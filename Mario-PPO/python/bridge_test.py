
import socket

HOST = "127.0.0.1"
PORT = 5555

print("Warte auf Java-Agent...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    connection, address = server.accept()

    print("Java verbunden:", address)

    with connection:
        with connection.makefile("r", encoding="utf-8") as stream:

            step = 0

            while True:

                line = stream.readline()

                if not line:
                    break

                parts = line.strip().split(",")

                if parts[0] == "END":
                    print("Spiel beendet.")
                    break

                if parts[0] != "OBS":
                    continue

                x = float(parts[1])
                y = float(parts[2])
                vx = float(parts[3])
                vy = float(parts[4])

                mode = int(parts[5])
                on_ground = parts[6].lower() == "true"
                can_jump = parts[7].lower() == "true"
                completion = float(parts[8])
                if step == 1:
                    print("Anzahl Grid-Werte:", len(grid))
                    print("Erste 30 Grid-Werte:", grid[:30])
                grid = [int(v) for v in parts[10:]]

                step += 1

                if step % 100 == 0:
                    print(
                        f"Step: {step} | "
                        f"X: {x:.1f} | "
                        f"Y: {y:.1f} | "
                        f"VX: {vx:.2f} | "
                        f"Ground: {on_ground} | "
                        f"Completion: {completion:.2%} | "
                        f"Grid: {len(grid)} Werte"
                    )

                # Testaktion: Mario läuft nach rechts
                action = 1

                connection.sendall(
                    f"{action}\n".encode("utf-8")
                )
print(f"Empfangene Spielschritte: {step}")
print("TCP-Test abgeschlossen.")
