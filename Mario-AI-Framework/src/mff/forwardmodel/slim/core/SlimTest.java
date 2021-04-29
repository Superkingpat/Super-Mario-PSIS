package mff.forwardmodel.slim.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class SlimTest {
    private static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException ignored) {
        }
        return content;
    }

    public static void main(String[] args) {
        //humanTest();
        agentTest();
    }

    private static void humanTest() {
        MarioGameSlim game = new MarioGameSlim();
        game.playGame(getLevel("./levels/original/lvl-1.txt"), 200, 0);
    }

    private static void agentTest() {
        /*for (int i = 1; i < 16; i++) {
            MarioGameSlim game = new MarioGameSlim();
            game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, true);
        }*/

        for (int i = 1; i < 16; i++) {
            double originalTime = 0;
            double slimTime = 0;

            for (int j = 0; j < 2; j++) {
                MarioGameSlim game = new MarioGameSlim();
                game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, true);
            }
            for (int k = 0; k < 10; k++) {
                MarioGameSlim game = new MarioGameSlim();
                TestResult testResult = game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, true);
                originalTime += testResult.originalTime;
                slimTime += testResult.slimTime;
            }

            System.out.println("Level: " + i + " original update time: " + originalTime);
            System.out.println("Level: " + i + " slim update time: " + slimTime);
            System.out.println();
            //System.out.print("Level " + i + ": ");
        }
    }
}
