package mff.forwardmodel.slim.core;

import mff.forwardmodel.slim.sprites.FireballSlim;
import mff.forwardmodel.slim.sprites.ShellSlim;

import java.util.ArrayList;
import java.util.LinkedList;

public class MarioUpdateContextSlim {

    public MarioWorldSlim world;
    public boolean[] actions;
    public int fireballsOnScreen;

    public final ArrayList<FireballSlim> fireballsToCheck = new ArrayList<>(2);
    public final ArrayList<ShellSlim> shellsToCheck = new ArrayList<>(2);
    final ArrayList<MarioSpriteSlim> addedSprites = new ArrayList<>(8);
    final ArrayList<MarioSpriteSlim> removedSprites = new ArrayList<>(8);

    private static final LinkedList<MarioUpdateContextSlim> pool = new LinkedList<>();

    public static MarioUpdateContextSlim get() {
        MarioUpdateContextSlim ctx = pool.poll();
        if (ctx != null) return ctx;
        return new MarioUpdateContextSlim();
    }

    static void back(MarioUpdateContextSlim ctx) {
        pool.add(ctx);
    }
}
