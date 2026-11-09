import java.nio.file.Files;
import java.nio.file.Path;

import engine.core.MarioGame;
import engine.core.MarioResult;

public class MarioPpoRunner {

    public static void main(String[] args) throws Exception {
        String levelPath = args.length > 0
            ? args[0]
            : "levels/original/lvl-1.txt";

        int timer = args.length > 1
            ? Integer.parseInt(args[1])
            : 20;

        int port = args.length > 2
            ? Integer.parseInt(args[2])
            : 5555;

        boolean visuals = args.length > 3
            ? Boolean.parseBoolean(args[3])
            : true;

        String level = Files.readString(Path.of(levelPath));
        MarioPpoAgent agent = new MarioPpoAgent("127.0.0.1", port);

        try {
            MarioGame game = new MarioGame();
            MarioResult result = game.runGame(
                agent,
                level,
                timer,
                0,
                visuals
            );

            agent.finish(result);
            PlayLevel.printResults(result);
        } finally {
            agent.close();
        }
    }
}
