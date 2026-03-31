package mff.python;

import engine.core.MarioGame;
import engine.core.MarioResult;
import mff.LevelLoader;

public class PythonControllerMain {
    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5050;
        String levelPath = args.length > 2 ? args[2] : "./levels/original/lvl-1.txt";
        int timer = args.length > 3 ? Integer.parseInt(args[3]) : 200;
        int marioState = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        boolean visuals = args.length > 5 ? Boolean.parseBoolean(args[5]) : true;

        String level = LevelLoader.getLevel(levelPath);
        MarioGame game = new MarioGame();

        PythonSocketAgent agent = new PythonSocketAgent(host, port);
        try {
            MarioResult result = game.runGame(agent, level, timer, marioState, visuals);
            agent.notifyGameOver(result);
            System.out.println("Game finished: " + result.getGameStatus());
            System.out.println("Completion: " + result.getCompletionPercentage());
            System.out.println("Remaining time: " + result.getRemainingTime());
        } finally {
            agent.close();
        }
    }
}