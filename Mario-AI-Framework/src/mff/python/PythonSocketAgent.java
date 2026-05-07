package mff.python;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioResult;
import engine.core.MarioTimer;
import engine.core.MarioWorld;
import engine.helper.MarioActions;
import mff.agents.common.IMarioAgentMFF;
import mff.agents.common.MarioTimerSlim;
import mff.forwardmodel.common.Converter;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public class PythonSocketAgent implements MarioAgent, AutoCloseable {
    private final String host;
    private final int port;
    private final int readTimeoutMs;

    private static final long ASTAR_ACTION_BUDGET_MS = 33;
    private static final int SLIM_CUTOUT_TILE_WIDTH = 27;

    private static final String ASTAR_ID_RB = "rb";
    private static final String ASTAR_ID_RB2009 = "rb2009";
    private static final String ASTAR_ID_MFF_ASTAR = "mff_astar";
    private static final String ASTAR_ID_MFF_ASTAR_PLANNING_DYNAMIC = "mff_astar_planning_dynamic";
    private static final String ASTAR_ID_MFF_ASTAR_WINDOW = "mff_astar_window";
    private static final String ASTAR_ID_MFF_RB_SLIM_WINDOW_ADVANCE = "mff_rb_slim_window_advance";


    private Socket socket;
    private BufferedReader reader;
    private BufferedWriter writer;
    private long stepCounter;

    public PythonSocketAgent(String host, int port) {
        this(host, port, 2000);
    }

    public PythonSocketAgent(String host, int port, int readTimeoutMs) {
        this.host = host;
        this.port = port;
        this.readTimeoutMs = readTimeoutMs;
    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        this.stepCounter = 0;
        connectIfNeeded();
        try {
            sendLine("HELLO\tMARIO_PY_BRIDGE\t1");
            sendLevelData(model);
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize Python socket agent", e);
        }
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        connectIfNeeded();
        stepCounter += 1;
        try {
            float[] pos = model.getMarioFloatPos();
            float[] vel = model.getMarioFloatVelocity();
            String enemies = serializeTriples(model.getEnemiesFloatPosAndType());
            String sprites = serializeTriples(model.getSpritesFloatPosAndType());
            String marioScene = serializeGrid(model.getMarioSceneObservation(0, 32, 16));
            String aStarActions = serializeAStarActions(model);
            String message = String.format(
                    Locale.US,
                    "STEP\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%b\t%b\t%b\t%d\t%.5f\t%s\t%s\t%s\t%s\t%s",
                    stepCounter,
                    pos[0],
                    pos[1],
                    vel[0],
                    vel[1],
                    model.getMarioMode(),
                    model.isMarioOnGround(),
                    model.mayMarioJump(),
                    model.getMarioCanJumpHigher(),
                    model.getRemainingTime(),
                    model.getCompletionPercentage(),
                    model.getGameStatus().name(),
                    enemies,
                    sprites,
                    marioScene,
                    aStarActions
            );
            sendLine(message);
            String response = reader.readLine();
            return parseActions(response);
        } catch (SocketTimeoutException e) {
            // Keep moving if the Python side times out.
            return defaultFallbackActions();
        } catch (IOException e) {
            throw new RuntimeException("Failed to communicate with Python controller", e);
        }
    }

    @Override
    public String getAgentName() {
        return "PythonSocketAgent";
    }

    public void notifyGameOver(MarioResult result) {
        if (!isConnected()) {
            return;
        }
        try {
            String message = String.format(
                    Locale.US,
                    "END\t%s\t%.5f\t%d\t%d\t%d\t%d",
                    result.getGameStatus().name(),
                    result.getCompletionPercentage(),
                    result.getRemainingTime(),
                    result.getNumJumps(),
                    result.getKillsTotal(),
                    result.getCurrentCoins()
            );
            sendLine(message);
        } catch (IOException ignored) {
            // Ignore cleanup errors.
        }
    }

    private void connectIfNeeded() {
        if (isConnected()) {
            return;
        }
        try {
            socket = new Socket(host, port);
            socket.setTcpNoDelay(true);
            socket.setSoTimeout(readTimeoutMs);
            reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException(
                    "Cannot connect to Python controller at " + host + ":" + port + ". Start PythonController/controller.py first.",
                    e
            );
        }
    }

    private boolean isConnected() {
        return socket != null && socket.isConnected() && !socket.isClosed();
    }

    private void sendLine(String line) throws IOException {
        writer.write(line);
        writer.newLine();
        writer.flush();
    }

    private boolean[] parseActions(String response) {
        boolean[] actions = new boolean[MarioActions.numberOfActions()];
        if (response == null || response.isBlank()) {
            return defaultFallbackActions();
        }

        String[] tokens = response.trim().split("[\\s,]+");
        if (tokens.length >= MarioActions.numberOfActions()) {
            for (int i = 0; i < MarioActions.numberOfActions(); i++) {
                actions[i] = isTruthy(tokens[i]);
            }
            return actions;
        }

        // Alternate format: action names, e.g. "RIGHT JUMP".
        for (String token : tokens) {
            String normalized = token.trim().toUpperCase(Locale.ROOT);
            if (normalized.equals("L")) actions[MarioActions.LEFT.getValue()] = true;
            if (normalized.equals("R")) actions[MarioActions.RIGHT.getValue()] = true;
            if (normalized.equals("D")) actions[MarioActions.DOWN.getValue()] = true;
            if (normalized.equals("S") || normalized.equals("RUN")) {
                actions[MarioActions.SPEED.getValue()] = true;
            }
            if (normalized.equals("J")) actions[MarioActions.JUMP.getValue()] = true;
        }
        return actions;
    }

    private String serializeAStarActions(MarioForwardModel model) {
        StringBuilder sb = new StringBuilder();
        appendAStarAction(sb, ASTAR_ID_RB, getRbAction(model, new agents.robinBaumgarten.Agent()));
        appendAStarAction(sb, ASTAR_ID_RB2009, getRb2009Action(model, new agents.robinBaumgarten2009.AStarAgent()));
        appendAStarAction(sb, ASTAR_ID_MFF_ASTAR, getMffAction(model, new mff.agents.astar.Agent()));
        appendAStarAction(sb, ASTAR_ID_MFF_ASTAR_PLANNING_DYNAMIC, getMffAction(model, new mff.agents.astarPlanningDynamic.Agent()));
        appendAStarAction(sb, ASTAR_ID_MFF_ASTAR_WINDOW, getMffAction(model, new mff.agents.astarWindow.Agent()));
        appendAStarAction(sb, ASTAR_ID_MFF_RB_SLIM_WINDOW_ADVANCE, getMffAction(model, new mff.agents.robinBaumgartenSlimWindowAdvance.Agent()));

        if (sb.length() == 0) {
            return "-";
        }
        return sb.toString();
    }

    private void appendAStarAction(StringBuilder sb, String id, boolean[] action) {
        if (action == null) {
            return;
        }
        if (sb.length() > 0) {
            sb.append(';');
        }
        sb.append(id).append('=').append(formatActionBits(action));
    }

    private String formatActionBits(boolean[] action) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < MarioActions.numberOfActions(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            boolean value = action != null && action.length > i && action[i];
            sb.append(value ? '1' : '0');
        }
        return sb.toString();
    }

    private boolean[] getRbAction(MarioForwardModel model, agents.robinBaumgarten.Agent agent) {
        MarioForwardModel snapshot = model.clone();
        try {
            agent.initialize(snapshot.clone(), new MarioTimer(ASTAR_ACTION_BUDGET_MS));
            return agent.getActions(snapshot, new MarioTimer(ASTAR_ACTION_BUDGET_MS));
        } catch (RuntimeException ex) {
            return new boolean[MarioActions.numberOfActions()];
        }
    }

    private boolean[] getRb2009Action(MarioForwardModel model, agents.robinBaumgarten2009.AStarAgent agent) {
        MarioForwardModel snapshot = model.clone();
        try {
            agent.initialize(snapshot.clone(), new MarioTimer(ASTAR_ACTION_BUDGET_MS));
            boolean[] raw = agent.getActions(snapshot, new MarioTimer(ASTAR_ACTION_BUDGET_MS));
            return normalizeRb2009Action(raw);
        } catch (RuntimeException ex) {
            return new boolean[MarioActions.numberOfActions()];
        }
    }

    private boolean[] normalizeRb2009Action(boolean[] raw) {
        boolean[] mapped = new boolean[MarioActions.numberOfActions()];
        if (raw == null) {
            return mapped;
        }
        int maxIndex = agents.robinBaumgarten2009.astar.sprites.Mario.KEY_SPEED;
        if (raw.length <= maxIndex) {
            return mapped;
        }
        mapped[MarioActions.LEFT.getValue()] = raw[agents.robinBaumgarten2009.astar.sprites.Mario.KEY_LEFT];
        mapped[MarioActions.RIGHT.getValue()] = raw[agents.robinBaumgarten2009.astar.sprites.Mario.KEY_RIGHT];
        mapped[MarioActions.DOWN.getValue()] = raw[agents.robinBaumgarten2009.astar.sprites.Mario.KEY_DOWN];
        mapped[MarioActions.JUMP.getValue()] = raw[agents.robinBaumgarten2009.astar.sprites.Mario.KEY_JUMP];
        mapped[MarioActions.SPEED.getValue()] = raw[agents.robinBaumgarten2009.astar.sprites.Mario.KEY_SPEED];
        return mapped;
    }

    private boolean[] getMffAction(MarioForwardModel model, IMarioAgentMFF agent) {
        try {
            MarioForwardModelSlim slimModel = Converter.originalToSlim(model.clone(), SLIM_CUTOUT_TILE_WIDTH);
            agent.initialize(slimModel.clone());
            return agent.getActions(slimModel.clone(), new MarioTimerSlim(ASTAR_ACTION_BUDGET_MS));
        } catch (RuntimeException ex) {
            return new boolean[MarioActions.numberOfActions()];
        }
    }

    private boolean isTruthy(String token) {
        String normalized = token.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("1")
                || normalized.equals("true")
                || normalized.equals("t")
                || normalized.equals("yes")
                || normalized.equals("y");
    }

    private boolean[] defaultFallbackActions() {
        boolean[] actions = new boolean[MarioActions.numberOfActions()];
        actions[MarioActions.RIGHT.getValue()] = true;
        actions[MarioActions.SPEED.getValue()] = true;
        return actions;
    }

    private String serializeTriples(float[] values) {
        if (values == null || values.length == 0) {
            return "-";
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i + 2 < values.length; i += 3) {
            if (sb.length() > 0) {
                sb.append(';');
            }
            sb.append(String.format(Locale.US, "%.0f,%.3f,%.3f", values[i], values[i + 1], values[i + 2]));
        }
        return sb.toString();
    }

    private String serializeGrid(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return "-";
        }

        int width = grid.length;
        int height = grid[0].length;
        StringBuilder sb = new StringBuilder();
        sb.append(width).append('x').append(height).append(':');

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (x != 0 || y != 0) {
                    sb.append(',');
                }
                sb.append(grid[x][y]);
            }
        }
        return sb.toString();
    }

    private void sendLevelData(MarioForwardModel model) throws IOException {
        MarioWorld world = model.getWorld();
        int[][] tiles = world.level.getLevelTiles();
        int width = tiles.length;
        int height = tiles[0].length;

        StringBuilder blocks = new StringBuilder();
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int tile = tiles[x][y];
                if (tile != 0) {
                    if (blocks.length() > 0) {
                        blocks.append(';');
                    }
                    blocks.append(x).append(',').append(y).append(',').append(tile);
                    }
                /*
                if (isInterestingBlock(tile)) {
                    
                    blocks.append(x).append(',').append(y).append(',').append(tile);
                }
                */
            }
        }

        sendLine("LEVEL\t" + width + "\t" + height + "\t" + (blocks.length() == 0 ? "-" : blocks));
    }

    private boolean isInterestingBlock(int tile) {
        return tile == 6   // breakable brick
                || tile == 7   // coin brick
                || tile == 8   // question block (special)
                || tile == 11  // question block (coin)
                || tile == 14  // used block
                || tile == 50  // special brick
                || tile == 51; // life brick
    }

    @Override
    public void close() {
        try {
            if (reader != null) {
                reader.close();
            }
        } catch (IOException ignored) {
        }
        try {
            if (writer != null) {
                writer.close();
            }
        } catch (IOException ignored) {
        }
        try {
            if (socket != null) {
                socket.close();
            }
        } catch (IOException ignored) {
        }
    }
}