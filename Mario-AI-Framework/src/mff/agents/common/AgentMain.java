package mff.agents.common;

import engine.core.MarioGame;
import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class AgentMain {
    private static String getLevel(String filepath) {
        System.out.println(filepath);
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
            return content;
        } catch (IOException ignored) {
            // try with working directory set one folder down
        }
        try {
            content = new String(Files.readAllBytes(Paths.get("." + filepath)));
        }
        catch (IOException e) {
            System.out.println("Level couldn't be loaded, please check the path provided with regards to your working directory.");
            System.exit(1);
        }
        return content;
    }

    private static IMarioAgentMFF getNewAgent(String agentType) {
        switch (agentType) {
            case "astar":
                return new mff.agents.astar.Agent();
            case "astarDistanceMetric":
                return new mff.agents.astarDistanceMetric.Agent();
            case "astarPlanningDynamic":
                return new mff.agents.astarPlanningDynamic.Agent();
            case "astarWindow":
                return new mff.agents.astarWindow.Agent();
            case "robinBaumgartenSlim":
                return new mff.agents.robinBaumgartenSlim.Agent();
            case "robinBaumgartenSlimImproved":
                return new mff.agents.robinBaumgartenSlimImproved.Agent();
            case "robinBaumgartenSlimWindowAdvance":
                return new mff.agents.robinBaumgartenSlimWindowAdvance.Agent();
            default:
                throw new IllegalArgumentException("Agent not supported.");
        }
    }

    public static void main(String[] args) {
        if (args[0].equals("robinBaumgarten")) {
            MarioGame game = new MarioGame();
            game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./" + args[1] + "/lvl-" + args[2] + ".txt"),
                    200, 0, true);
        }
        else
            testLevel(getNewAgent(args[0]), args[1], args[2]);
        //testAllOriginalLevels();
        //testGeneratedLevels();
        //testAllAgents();
    }

    private static void testLevel(IMarioAgentMFF agent, String levelType, String levelNum) {
        AgentMarioGame game = new AgentMarioGame();
        game.runGame(agent, getLevel("./" + levelType + "/lvl-" + levelNum + ".txt"),
                200, 0, true);
    }

    private static void testAllOriginalLevels() {
        for (int i = 1; i < 16; i++) {
            AgentMarioGame game = new AgentMarioGame();
            game.runGame(new mff.agents.robinBaumgartenSlimWindowAdvance.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"),
                    200, 0, true);
        }
    }

    private static void testGeneratedLevels() {
        for (int i = 1; i <= 100; i++) {
            MarioLevelGenerator generator = new levelGenerators.krys.LevelGenerator(i);
            String level = generator.getGeneratedLevel(new MarioLevelModel(150, 16),
                    new MarioTimer(5 * 60 * 60 * 1000));
            AgentMarioGame game = new AgentMarioGame();
            game.runGame(new mff.agents.astarWindow.Agent(), level, 30, 0, false);
        }
    }

    private static void testAllAgents() {
        ArrayList<IMarioAgentMFF> agents = new ArrayList<>() {{
            add(new mff.agents.astar.Agent());
            add(new mff.agents.astarDistanceMetric.Agent());
            add(new mff.agents.astarFast.Agent());
            add(new mff.agents.astarJump.Agent());
            add(new mff.agents.astarPlanning.Agent());
            add(new mff.agents.astarPlanningDynamic.Agent());
            add(new mff.agents.astarWindow.Agent());
            add(new mff.agents.robinBaumgartenSlim.Agent());
            add(new mff.agents.robinBaumgartenSlimImproved.Agent());
        }};

        for (var agent : agents) {
            AgentMarioGame game = new AgentMarioGame();
            System.out.println("Testing " + agent.getAgentName());
            game.runGame(agent, getLevel("./levels/original/lvl-1.txt"), 200, 0, true);
        }
    }
}
