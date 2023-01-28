package mff.agents.benchmark;

import engine.core.MarioAgent;
import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;
import mff.agents.astar.AStarTree;
import mff.agents.common.IMarioAgentMFF;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Locale;

public class AgentBenchmark {

    private static final DecimalFormat twoFractionDigitsCommaSeparator;

    static {
        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.getDefault());
        symbols.setDecimalSeparator('.');
        twoFractionDigitsCommaSeparator = new DecimalFormat("0.00", symbols);
    }

    private static final ArrayList<String> agents = new ArrayList<>() {{
//        add("robinBaumgarten");
//        add("robinBaumgartenSlimWindowAdvance");
        add("astar");
//        add("astarPlanningDynamic");
//        add("astarWindow");
    }};

    private static final ArrayList<String> levels = new ArrayList<>() {{
        add("ge");
        add("hopper");
        add("notch");
        add("notchParam");
        add("notchParamRand");
        add("ore");
        add("patternCount");
        add("patternOccur");
        add("patternWeightCount");
    }};

    public static void main(String[] args) throws IOException {
        try {
            AStarTree.nodeDepthWeight = Float.parseFloat(args[0]);
            AStarTree.timeToFinishWeight = Float.parseFloat(args[1]);
        } catch (Exception e) {
            System.out.println("nodeDepthWeight or timeToFinishWeight not set successfully");
            throw e;
        }
//        for (String level : levels) {
            for (var agentType : agents) {
                File log = prepareLog("agent-benchmark" + File.separator + agentType + "-krys-"
                        + "NDW-" + AStarTree.nodeDepthWeight + "-TTFW-" + AStarTree.timeToFinishWeight + ".csv");
                if (log == null)
                    return;
                FileWriter logWriter = new FileWriter(log);

//                logWriter.write("running with node depth weight: " + AStarTree.nodeDepthWeight + " and time to finish weight: " + AStarTree.timeToFinishWeight + "\n");
                logWriter.write("ndw:" + AStarTree.nodeDepthWeight + "\n");
                logWriter.write("ttfw:" + AStarTree.timeToFinishWeight + "\n");
                logWriter.write("level,win/fail,% travelled,run time,game ticks,planning time,total plannings,nodes evaluated\n");

                warmup(agentType);
//                testOriginalLevels(agentType, logWriter);
                testKrysLevels(agentType, logWriter);
//                testFrameworkLevels(agentType, logWriter, level);

                logWriter.close();
            }
//        }
    }

    private static void testFrameworkLevels(String agentType, FileWriter log, String levelsName) throws IOException {
        AgentStats agentStats;
        if (!agentType.equals("robinBaumgarten")) {
            for (int i = 1; i <= 100; i++) {
                System.out.println(agentType + "-" + levelsName + "-" + i);
                String level = getLevel("./levels/" + levelsName + "/lvl-" + i + ".txt");
                AgentBenchmarkGame game = new AgentBenchmarkGame();
                IMarioAgentMFF agent = getNewAgent(agentType);
                // only 30 seconds to speed-up timeout if agent is stuck
                agentStats = game.runGame(agent, level, 30, 0, false);
                agentStats.level = levelsName + "-" + i;
                printStats(log, agentStats);
            }
        }
        else {
            for (int i = 1; i <= 100; i++) {
                System.out.println(agentType + "-" + levelsName + "-" + i);
                String level = getLevel("./levels/" + levelsName + "/lvl-" + i + ".txt");
                OriginalAgentBenchmarkGame game = new OriginalAgentBenchmarkGame();
                MarioAgent agent = new agents.robinBaumgarten.Agent();
                // only 30 seconds to speed-up timeout if agent is stuck
                agentStats = game.runGame(agent, level, 30, 0, false);
                agentStats.level = levelsName + "-" + i;
                printStats(log, agentStats);
            }
        }
    }

    private static void testKrysLevels(String agentType, FileWriter log) throws IOException {
        AgentStats agentStats;
        if (!agentType.equals("robinBaumgarten")) {
            for (int i = 1; i <= 100; i++) {
                System.out.println(agentType + "-" + "krys" + "-" + i);
//                MarioLevelGenerator generator = new levelGenerators.krys.LevelGenerator(i);
//                String level = generator.getGeneratedLevel(new MarioLevelModel(150, 16),
//                        new MarioTimer(5 * 60 * 60 * 1000));
                String level = getLevel("./levels/krys/lvl-" + i + ".txt");
                AgentBenchmarkGame game = new AgentBenchmarkGame();
                IMarioAgentMFF agent = getNewAgent(agentType);
                // only 30 seconds to speed-up timeout if agent is stuck
                agentStats = game.runGame(agent, level, 30, 0, false);
                agentStats.level = "Krys-" + i;
                printStats(log, agentStats);
            }
        }
        else {
            for (int i = 1; i <= 100; i++) {
                System.out.println(agentType + "-" + "krys" + "-" + i);
                MarioLevelGenerator generator = new levelGenerators.krys.LevelGenerator(i);
                String level = generator.getGeneratedLevel(new MarioLevelModel(150, 16),
                        new MarioTimer(5 * 60 * 60 * 1000));
                OriginalAgentBenchmarkGame game = new OriginalAgentBenchmarkGame();
                MarioAgent agent = new agents.robinBaumgarten.Agent();
                // only 30 seconds to speed-up timeout if agent is stuck
                agentStats = game.runGame(agent, level, 30, 0, false);
                agentStats.level = "Krys-" + i;
                printStats(log, agentStats);
            }
        }
    }

    private static void testOriginalLevels(String agentType, FileWriter log) throws IOException {
        AgentStats agentStats;
        if (!agentType.equals("robinBaumgarten")) {
            for (int i = 1; i < 16; i++) {
                System.out.println(agentType + "-" + "original" + "-" + i);
                AgentBenchmarkGame game = new AgentBenchmarkGame();
                IMarioAgentMFF agent = getNewAgent(agentType);
                String level = getLevel("./levels/original/lvl-" + i + ".txt");
                agentStats = game.runGame(agent, level,200, 0, false);
                agentStats.level = "Mario-" + i;
                printStats(log, agentStats);
            }
        }
        else {
            for (int i = 1; i < 16; i++) {
                System.out.println(agentType + "-" + "original" + "-" + i);
                OriginalAgentBenchmarkGame game = new OriginalAgentBenchmarkGame();
                MarioAgent agent = new agents.robinBaumgarten.Agent();
                String level = getLevel("./levels/original/lvl-" + i + ".txt");
                agentStats = game.runGame(agent, level, 200, 0, false);
                agentStats.level = "Mario-" + i;
                printStats(log, agentStats);
            }
        }
    }

    private static void warmup(String agentType) {
        System.out.println("WARMUP: " + agentType + "-" + "original-1");
        if (!agentType.equals("robinBaumgarten")) {
            AgentBenchmarkGame game = new AgentBenchmarkGame();
            IMarioAgentMFF agent = getNewAgent(agentType);
            String level = getLevel("./levels/original/lvl-1.txt");
            game.runGame(agent, level,200, 0, false);
        }
        else {
            OriginalAgentBenchmarkGame game = new OriginalAgentBenchmarkGame();
            MarioAgent agent = new agents.robinBaumgarten.Agent();
            String level = getLevel("./levels/original/lvl-1.txt");
            game.runGame(agent, level, 200, 0, false);
        }
    }

    private static void printStats(FileWriter writer, AgentStats stats) throws IOException {
        writer.write(stats.level + ','
                + stats.win + ','
                + twoFractionDigitsCommaSeparator.format(stats.percentageTravelled) + ','
                + stats.runTime + ','
                + stats.totalGameTicks + ','
                + stats.totalPlanningTime + ','
                + stats.searchCalls + ','
                + stats.nodesEvaluated + '\n'
        );
    }

    private static File prepareLog(String name) throws IOException {
        File agentBenchmarkFolder = new File("agent-benchmark");
        if (!agentBenchmarkFolder.exists()) {
            if (!agentBenchmarkFolder.mkdir()) {
                System.out.println("Can't create folder: " + agentBenchmarkFolder.getName());
                return null;
            }
        }

        File log = new File(name);
        if (log.exists()) {
            if (!log.delete()) {
                System.out.println("Can't delete file: " + log.getName());
                return null;
            }
        }
        if (!log.createNewFile()) {
            System.out.println("Can't create file: " + log.getName());
            return null;
        }
        return log;
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

    private static String getLevel(String filepath) {
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
}
