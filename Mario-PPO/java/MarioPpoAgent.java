import java.io.*;
import java.net.Socket;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioResult;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

public class MarioPpoAgent implements MarioAgent, AutoCloseable {

    private final String host;
    private final int port;

    private Socket socket;
    private BufferedReader input;
    private PrintWriter output;

    public MarioPpoAgent(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        try {
            socket = new Socket(host, port);
            socket.setTcpNoDelay(true);

            input = new BufferedReader(
                new InputStreamReader(socket.getInputStream())
            );
            output = new PrintWriter(socket.getOutputStream(), true);

            System.out.println("Python-Verbindung hergestellt.");
        } catch (IOException e) {
            throw new RuntimeException("Verbindung zu Python fehlgeschlagen", e);
        }
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        try {
            float[] pos = model.getMarioFloatPos();
            float[] vel = model.getMarioFloatVelocity();
            int[][] grid = model.getMarioCompleteObservation();

            StringBuilder message = new StringBuilder("OBS");
            message.append(",").append(pos[0]);
            message.append(",").append(pos[1]);
            message.append(",").append(vel[0]);
            message.append(",").append(vel[1]);
            message.append(",").append(model.getMarioMode());
            message.append(",").append(model.isMarioOnGround());
            message.append(",").append(model.mayMarioJump());
            message.append(",").append(model.getCompletionPercentage());
            message.append(",").append(model.getRemainingTime());

            for (int[] row : grid) {
                for (int value : row) {
                    message.append(",").append(value);
                }
            }

            output.println(message);

            String response = input.readLine();
            if (response == null) {
                throw new IOException("Python-Verbindung beendet");
            }

            int action = Integer.parseInt(response.trim());
            return decodeAction(action);

        } catch (Exception e) {
            throw new RuntimeException("Fehler bei der Agent-Kommunikation", e);
        }
    }

    // Aktion: 0=keine, 1=rechts, 2=rechts+schnell, 3=rechts+Sprung,
    // 4=rechts+schnell+Sprung, 5=Sprung, 6=schnell+Sprung,
    // 7=links, 8=links+schnell, 9=links+Sprung,
    // 10=links+schnell+Sprung, 11=schnell
    private boolean[] decodeAction(int action) {
        boolean[] buttons =
            new boolean[MarioActions.numberOfActions()];

        switch (action) {
            case 0:
                break;
            case 1:
                buttons[1] = true;
                break;
            case 2:
                buttons[1] = true;
                buttons[3] = true;
                break;
            case 3:
                buttons[1] = true;
                buttons[4] = true;
                break;
            case 4:
                buttons[1] = true;
                buttons[3] = true;
                buttons[4] = true;
                break;
            case 5:
                buttons[4] = true;
                break;
            case 6:
                buttons[3] = true;
                buttons[4] = true;
                break;
            case 7:
                buttons[0] = true;
                break;
            case 8:
                buttons[0] = true;
                buttons[3] = true;
                break;
            case 9:
                buttons[0] = true;
                buttons[4] = true;
                break;
            case 10:
                buttons[0] = true;
                buttons[3] = true;
                buttons[4] = true;
                break;
            case 11:
                buttons[3] = true;
                break;
            default:
                throw new IllegalArgumentException("Unbekannte Aktion: " + action);
        }

        return buttons;
    }

    public void finish(MarioResult result) {
        if (output != null) {
            output.println(
                "RESULT," + result.getCompletionPercentage()
                + "," + result.getGameStatus().toString()
            );
        }
    }

    @Override
    public String getAgentName() {
        return "Python PPO Agent";
    }

    @Override
    public void close() {
        try {
            if (output != null) {
                output.println("END");
            }
            if (socket != null) {
                socket.close();
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }
}
