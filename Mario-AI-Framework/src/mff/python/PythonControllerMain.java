package mff.python;

import engine.core.MarioGame;
import engine.core.MarioResult;
import mff.LevelLoader;

public class PythonControllerMain {
    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5050;
        String levelPathArg = args.length > 2 ? args[2] : "./levels/original/lvl-1.txt";
        int timer = args.length > 3 ? Integer.parseInt(args[3]) : 200;
        int marioState = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        boolean visuals = args.length > 5 ? Boolean.parseBoolean(args[5]) : true;
        int sessions = args.length > 6 ? Integer.parseInt(args[6]) : 1;
        int sessionTimeoutSeconds = args.length > 7 ? Integer.parseInt(args[7]) : -1;

        String[] levelPaths = levelPathArg.split(";");
        if (levelPaths.length == 0) {
            levelPaths = new String[]{"./levels/original/lvl-1.txt"};
        }

        MarioGame game = new MarioGame();
        PythonSocketAgent agent = new PythonSocketAgent(host, port);
        try {
            for (int i = 0; i < sessions; i++) {
                String levelPath = levelPaths[i % levelPaths.length];
                String level = LevelLoader.getLevel(levelPath);

                System.out.println("Session " + (i + 1) + "/" + sessions + " - Level: " + levelPath);

                if (sessionTimeoutSeconds > 0) {
                    game.setWallClockTimeoutMs(sessionTimeoutSeconds * 1000L);
                } else {
                    game.setWallClockTimeoutMs(-1);
                }

                MarioResult result = game.runGame(agent, level, timer, marioState, visuals);
                agent.notifyGameOver(result);
                System.out.println("Game finished: " + result.getGameStatus());
                System.out.println("Completion: " + result.getCompletionPercentage());
                System.out.println("Remaining time: " + result.getRemainingTime());
            }
        } finally {
            agent.close();
        }
    }
}