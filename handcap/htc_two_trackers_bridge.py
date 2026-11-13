import argparse
import json
import socket
import sys
import time
from pathlib import Path

from htc_control_core import HTCRemoteController, unix_time_ms

MDNS_ADDR = "224.0.0.251"
MDNS_PORT = 5353
INSTAUMI_SERVICE_TYPE = "_instaumi-htc._tcp.local."


def send_json_line(sock, payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sock.sendall(encoded + b"\n")


def recv_json_lines(sock, buffer):
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        if not line:
            continue
        try:
            yield json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON from iPad: {exc}")
    return buffer


def encode_dns_name(name):
    labels = name.rstrip(".").split(".")
    encoded = bytearray()
    for label in labels:
        raw = label.encode("utf-8")
        encoded.append(len(raw))
        encoded.extend(raw)
    encoded.append(0)
    return bytes(encoded)


def decode_dns_name(packet, offset):
    labels = []
    jumped = False
    next_offset = offset
    seen = set()

    while True:
        if offset >= len(packet):
            raise ValueError("DNS name offset out of range")
        length = packet[offset]
        if length == 0:
            offset += 1
            if not jumped:
                next_offset = offset
            break
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(packet):
                raise ValueError("DNS compression pointer out of range")
            pointer = ((length & 0x3F) << 8) | packet[offset + 1]
            if pointer in seen:
                raise ValueError("DNS compression pointer loop")
            seen.add(pointer)
            if not jumped:
                next_offset = offset + 2
            offset = pointer
            jumped = True
            continue

        offset += 1
        label = packet[offset : offset + length].decode("utf-8", errors="replace")
        labels.append(label)
        offset += length
        if not jumped:
            next_offset = offset

    return ".".join(labels) + ".", next_offset


def build_mdns_query(name, qtype):
    header = b"\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
    question = encode_dns_name(name) + qtype.to_bytes(2, "big") + (0x8001).to_bytes(2, "big")
    return header + question


def parse_mdns_records(packet):
    if len(packet) < 12:
        return []

    qdcount = int.from_bytes(packet[4:6], "big")
    ancount = int.from_bytes(packet[6:8], "big")
    nscount = int.from_bytes(packet[8:10], "big")
    arcount = int.from_bytes(packet[10:12], "big")
    offset = 12

    try:
        for _ in range(qdcount):
            _, offset = decode_dns_name(packet, offset)
            offset += 4

        records = []
        for _ in range(ancount + nscount + arcount):
            name, offset = decode_dns_name(packet, offset)
            if offset + 10 > len(packet):
                break
            rtype = int.from_bytes(packet[offset : offset + 2], "big")
            rclass = int.from_bytes(packet[offset + 2 : offset + 4], "big") & 0x7FFF
            ttl = int.from_bytes(packet[offset + 4 : offset + 8], "big")
            rdlength = int.from_bytes(packet[offset + 8 : offset + 10], "big")
            offset += 10
            rdata_offset = offset
            rdata = packet[offset : offset + rdlength]
            offset += rdlength

            value = None
            if rclass != 1:
                continue
            if rtype == 1 and rdlength == 4:
                value = socket.inet_ntoa(rdata)
            elif rtype == 12:
                value, _ = decode_dns_name(packet, rdata_offset)
            elif rtype == 33 and rdlength >= 6:
                port = int.from_bytes(packet[rdata_offset + 4 : rdata_offset + 6], "big")
                target, _ = decode_dns_name(packet, rdata_offset + 6)
                value = {"port": port, "target": target}

            if value is not None:
                records.append({"name": name, "type": rtype, "ttl": ttl, "value": value})
        return records
    except (IndexError, ValueError):
        return []


def discover_ipad_control_service(service_type=INSTAUMI_SERVICE_TYPE, timeout=4.0):
    deadline = time.time() + timeout
    instances = set()
    srv_by_instance = {}
    addresses_by_target = {}

    queries = [
        build_mdns_query(service_type, 12),
        build_mdns_query(service_type, 33),
    ]

    print(f"Discovering iPad control service via Bonjour/mDNS: {service_type}")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.settimeout(0.4)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)

        for query in queries:
            sock.sendto(query, (MDNS_ADDR, MDNS_PORT))

        while time.time() < deadline:
            try:
                packet, _ = sock.recvfrom(9000)
            except socket.timeout:
                continue

            for record in parse_mdns_records(packet):
                rtype = record["type"]
                name = record["name"]
                value = record["value"]
                if rtype == 12 and name.lower() == service_type.lower():
                    instances.add(value)
                    sock.sendto(build_mdns_query(value, 33), (MDNS_ADDR, MDNS_PORT))
                elif rtype == 33:
                    instances.add(name)
                    srv_by_instance[name] = value
                    sock.sendto(build_mdns_query(value["target"], 1), (MDNS_ADDR, MDNS_PORT))
                elif rtype == 1:
                    addresses_by_target.setdefault(name, []).append(value)

            for instance in instances:
                srv = srv_by_instance.get(instance)
                if not srv:
                    continue
                addresses = addresses_by_target.get(srv["target"], [])
                if addresses:
                    return addresses[0], srv["port"], instance

    return None


class HTCUSBBridge:
    def __init__(self, host, port, controller, reconnect_delay):
        self.host = host
        self.port = port
        self.controller = controller
        self.reconnect_delay = reconnect_delay
        self.failure_count = 0

    def run_forever(self):
        while True:
            try:
                self.run_once()
                self.failure_count = 0
            except KeyboardInterrupt:
                print("Stopping HTC two-trackers bridge.")
                return
            except OSError as exc:
                self.failure_count += 1
                print(f"Control bridge disconnected or unavailable: {exc}")
                if self.failure_count in (3, 10) or self.failure_count % 30 == 0:
                    self.print_troubleshooting()
                time.sleep(self.reconnect_delay)

    def print_troubleshooting(self):
        print("Still waiting for the iPad control socket.")
        print("Check these in order:")
        print("  1. iPad is connected by USB forwarding, or Windows can reach the iPad IP over LAN.")
        print("  2. instaumi is open on the iPad, on 数据采集, with Windows HTC 控制 enabled.")
        print(f"  3. The app USB/LAN port is {self.port}.")
        print(f"  4. USB mode: the forward window is still running: iproxy {self.port} {self.port}")
        print("  5. LAN mode: use --host <iPad IP>, and make sure firewall/network allows the port.")

    def run_once(self):
        print(f"Connecting to iPad control socket at {self.host}:{self.port} ...")
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.settimeout(None)
            print("Connected to iPad control socket.")
            self.send_hello(sock)

            buffer = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("iPad closed the control socket")
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    if raw_line:
                        self.handle_message(sock, raw_line)

    def send_hello(self, sock):
        now = unix_time_ms()
        status = self.controller.status()
        status.update(
            {
                "type": "hello",
                "message": "Windows control bridge ready",
                "server_receive_unix_time_ms": now,
                "server_send_unix_time_ms": now,
                "server_unix_time_ms": now,
            }
        )
        send_json_line(sock, status)

    def handle_message(self, sock, raw_line):
        receive_ms = unix_time_ms()
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            send_json_line(
                sock,
                {
                    "ok": False,
                    "message": f"Invalid JSON: {exc}",
                    "server_receive_unix_time_ms": receive_ms,
                    "server_send_unix_time_ms": unix_time_ms(),
                    "server_unix_time_ms": unix_time_ms(),
                },
            )
            return

        command = payload.get("command")
        request_id = payload.get("request_id")

        if command == "ping":
            result = self.controller.status()
            result["message"] = "Windows control bridge ready"
        elif command == "start":
            result = self.controller.start(payload, receive_ms)
        elif command == "stop":
            result = self.controller.stop(payload, receive_ms)
        else:
            result = {
                "ok": False,
                "message": f"Unknown USB command: {command}",
                "running": self.controller.status().get("running", False),
                "pid": self.controller.status().get("pid"),
            }

        send_ms = unix_time_ms()
        result.update(
            {
                "request_id": request_id,
                "server_receive_unix_time_ms": receive_ms,
                "server_send_unix_time_ms": send_ms,
                "server_unix_time_ms": send_ms,
            }
        )
        send_json_line(sock, result)


def main():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="TCP bridge for iPad-controlled htc_two_trackers.py; use USB forwarding or LAN")
    parser.add_argument("--host", help="iPad hostname or IP, or omit for USB forwarding")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--discover", action="store_true", help="Discover the iPad app with Bonjour/mDNS before connecting")
    parser.add_argument("--discover-timeout", type=float, default=4.0)
    parser.add_argument("--service-type", default=INSTAUMI_SERVICE_TYPE)
    parser.add_argument("--cap-id", type=int, default=112501)
    parser.add_argument("--collect-on-start", action="store_true", default=True)
    parser.add_argument("--no-collect-on-start", action="store_false", dest="collect_on_start")
    parser.add_argument("--script", default=str(root_dir / "htc_two_trackers.py"))
    parser.add_argument("--log-dir", default=str(root_dir / "remote_control_two_trackers_logs"))
    parser.add_argument("--reconnect-delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.discover:
        discovered = discover_ipad_control_service(args.service_type, timeout=args.discover_timeout)
        if discovered:
            host, port, instance = discovered
            args.host = host
            args.port = port
            print(f"Discovered {instance} at {args.host}:{args.port}")
        elif args.host:
            print(f"Bonjour discovery did not find the iPad app; falling back to {args.host}:{args.port}")
        else:
            parser.error("Bonjour discovery did not find the iPad app. Supply its IP with --host.")

    args.host = args.host or "127.0.0.1"

    controller = HTCRemoteController(
        root_dir=root_dir,
        script_path=args.script,
        log_dir=args.log_dir,
        default_cap_id=args.cap_id,
        default_collect_on_start=args.collect_on_start,
    )

    print("HTC two-trackers control bridge")
    print(f"Python: {sys.executable}")
    print(f"Script: {Path(args.script).resolve()}")
    print(f"Logs: {Path(args.log_dir).resolve()}")
    if args.host in {"127.0.0.1", "localhost"}:
        print("Expected USB forward: iproxy {0} {0}".format(args.port))
    else:
        print(f"LAN mode: connecting directly to iPad at {args.host}:{args.port}")

    bridge = HTCUSBBridge(args.host, args.port, controller, args.reconnect_delay)
    bridge.run_forever()


if __name__ == "__main__":
    main()
