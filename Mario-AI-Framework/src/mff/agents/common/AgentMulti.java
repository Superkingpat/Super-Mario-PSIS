package mff.agents.common;

import mff.LevelLoader;

import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AgentMulti {
    private static final int FRAME_HORIZONTAL_PADDING = 16;
    private static final int FRAME_VERTICAL_PADDING = 40;

    public static void main(String[] args) {
        if (args.length > 0 && "--single".equals(args[0])) {
            runSingle(args);
            return;
        }

        tileOriginalLevels();
    }

    private static void tileOriginalLevels() {
        String[] levels = new String[]{
                "./levels/original/lvl-1.txt",
                "./levels/original/lvl-2.txt",
                "./levels/original/lvl-3.txt",
                "./levels/original/lvl-4.txt"
        };

        tileLevels(levels, 2f, 200, 0);
    }

    private static void tileLevels(String[] levelPaths, float scale, int timer, int marioState) {
        if (levelPaths == null || levelPaths.length == 0) {
            return;
        }

        int levelCount = levelPaths.length;
        int cols = (int) Math.ceil(Math.sqrt(levelCount));
        int tileWidth = Math.round(AgentMarioGame.width * scale) + FRAME_HORIZONTAL_PADDING;
        int tileHeight = Math.round(AgentMarioGame.height * scale) + FRAME_VERTICAL_PADDING;

        for (int i = 0; i < levelCount; i++) {
            String levelPath = levelPaths[i];
            int col = i % cols;
            int row = i / cols;
            Point windowPos = new Point(col * tileWidth, row * tileHeight);
            startProcess(levelPath, scale, timer, marioState, windowPos.x, windowPos.y);
        }
    }

    private static void runSingle(String[] args) {
        if (args.length < 7) {
            throw new IllegalArgumentException("Expected: --single <levelPath> <x> <y> <scale> <timer> <marioState>");
        }

        String levelPath = args[1];
        int x = Integer.parseInt(args[2]);
        int y = Integer.parseInt(args[3]);
        float scale = Float.parseFloat(args[4]);
        int timer = Integer.parseInt(args[5]);
        int marioState = Integer.parseInt(args[6]);

        AgentMarioGame game = new AgentMarioGame();
        String title = "Mario AI Framework - " + levelPath;
        game.runGame(new mff.agents.astar.Agent(), LevelLoader.getLevel(levelPath),
                timer, marioState, true, 30, scale, title, new Point(x, y));
    }

    private static void startProcess(String levelPath, float scale, int timer, int marioState, int x, int y) {
        String javaExecutable = System.getProperty("java.home") + "/bin/java";
        String classPath = System.getProperty("java.class.path");

        List<String> command = new ArrayList<>();
        command.add(javaExecutable);
        command.add("-cp");
        command.add(classPath);
        command.add(AgentMulti.class.getName());
        command.add("--single");
        command.add(levelPath);
        command.add(Integer.toString(x));
        command.add(Integer.toString(y));
        command.add(Float.toString(scale));
        command.add(Integer.toString(timer));
        command.add(Integer.toString(marioState));

        try {
            new ProcessBuilder(command)
                    .inheritIO()
                    .start();
        } catch (IOException e) {
            throw new RuntimeException("Failed to start tiled process for level: " + levelPath, e);
        }
    }
}
