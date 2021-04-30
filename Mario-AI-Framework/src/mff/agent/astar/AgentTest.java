package mff.agent.astar;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import mff.agent.helper.AgentMarioGame;

public class AgentTest {
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
        tesAllOriginalLevels();
    }

    private static void testLevel() {
        AgentMarioGame game = new AgentMarioGame();
        game.runGame(new Agent(), getLevel("./levels/original/lvl-1.txt"), 200, 0, true);
    }
//TODO: ZIG ZAG is in level 4, works with enough time
//TODO: no action?
//TODO: y difference?
//TODO: use time in cost?
//TODO: beginning of level 3
//TODO: more time helps a lot
    private static void tesAllOriginalLevels() {
        for (int i = 1; i < 16; i++) {
            AgentMarioGame game = new AgentMarioGame();
            System.out.print("Level " + i + ": ");
            game.runGame(new mff.agent.astar.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, true);
        }
    }
}
