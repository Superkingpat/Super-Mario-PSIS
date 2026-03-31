package mff.python;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioResult;
import engine.core.MarioTimer;
import engine.core.MarioWorld;
import engine.helper.MarioActions;

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
                String marioScene = serializeGrid(model.getMarioSceneObservation(0));
            String message = String.format(
                    Locale.US,
                    "STEP\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%b\t%b\t%b\t%d\t%.5f\t%s\t%s\t%s\t%s",
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
                    marioScene
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
                    "END\t%s\t%.5f\t%d\t%d\t%d",
                    result.getGameStatus().name(),
                    result.getCompletionPercentage(),
                    result.getRemainingTime(),
                    result.getNumJumps(),
                    result.getKillsTotal()
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