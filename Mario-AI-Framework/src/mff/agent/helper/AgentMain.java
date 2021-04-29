package mff.agent.helper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class AgentMain {
    private static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException ignored) {
        }
        return content;
    }

    public static void main(String[] args) {
        //testLevel();
        testAllOriginalLevels();
    }

    private static void testLevel() {
        AgentMarioGame game = new AgentMarioGame();
        game.runGame(new mff.agent.core.Agent(), getLevel("./levels/original/lvl-1.txt"), 200, 0, true);
    }

    private static void testAllOriginalLevels() {
        for (int i = 1; i < 16; i++) {
            AgentMarioGame game = new AgentMarioGame();
            System.out.print("Level " + i + ": ");
            game.runGame(new mff.agent.core.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, true);
        }
    }
}
