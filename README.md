### Related papers
[Super Mario A-Star Agent Reloaded](https://ieeexplore.ieee.org/document/11272516) [(DOI)](https://doi.org/10.1109/ICTAI66417.2025.00190)

[Super Mario A-Star Agent Revisited](https://ieeexplore.ieee.org/document/9643319) [(DOI)](https://doi.org/10.1109/ICTAI52525.2021.00161)

### Introduction
This project is based on the framework created by [Ahmed Khalifa](https://scholar.google.com/citations?user=DRcyg5kAAAAJ&hl=en), which can be found [here](https://github.com/amidos2006/Mario-AI-Framework). As a part of my bachelor thesis, I created a better forward model for this framework, and to prove its functionality, I built a few intelligent agents on top of it.

The work on this project was continued as a part of my diploma thesis, where I created even better agents, especially the `astarGrid` agent (also referred to as `MFF A* Grid`) and the `astarWaypoints` (`MFF A* Waypoints`) agent.

### Requirements
I tried to support different working directory settings of various IDEs. It is guaranteed to work with working directory set in the `Mario-AI-Framework` folder, which contains folders such as `src` and `levels`. Java OpenJDK 17 might be required, but the framework probably runs on earlier versions too.

### Project overview

- `agent-benchmark` - an output folder for agents' benchmark results (created by the program when a benchmark is run)
- `img` - graphical assets of the game
- `levels` - original, generated and test levels
- `src` - source files of the framework
  - `agents` - agents from the framework mostly created during competitions
  - `engine` - original game implementation and forward model
  - `levelGenerators`
    -  generators from the framework mostly created during competitions
    - the `krys` and `noiseBased` generators were created by MFF UK students Jan Holan and Mikuláš Hrdlička as a part of the Procedural Content Generation course
  - `mff` - the source code of our works
    - `agents` - all of the agents + a benchmark environment for them; also contains an implementation of the grid search
    - `forwardmodel` - contains the two new forward models
      - `slim` - which is an improved version of the original forward model
      - `bin` - which is an experimental model that isn't finished

### Experiment results

Some of the experiment results (those that could be reasonably uploaded) are present in the ` Experiment-Results` folder.

### Interesting entry points

- `src/mff/HumanPlaytesting.java`, which allows you to play any level you want manually.

- `src/mff/agents/common/AgentMain.java`, which shows how to set up and run various agents on a given level or a set of levels, including the `MFF A*`, `MFF A* Grid` and `MFF A* Waypoints` agents; it can also be used to ensure that the benchmark class correctly collects data.

- `src/mff/agents/gridSearch/GridSearchMain.java`, which allows the visualisation of grid path.

- `src/mff/agents/benchmark/AgentBenchmarkMetacentrum.java`, which is used for the parameter search for `MFF A* Grid`, be warned that this class requires command line parameters, please refer to the `Parameter search` section before using this class.

- `src/mff/agents/benchmark/GridSearchBenchmark.java`, which is a simple benchmark of the performance of the grid search.

### How to run

In the `Mario-AI-Framework` folder, run:

- javac -cp src `your-desired-entry-point`
  - e.g. javac -cp src src/mff/agents/common/AgentMain.java

followed by:

- java -cp src `your-desired-entry-point`.java
  - e.g. java -cp src src/mff/agents/common/AgentMain.java

this particular example runs the file containing various example agent setups, please check the file and uncomment the method that you would like to try.

### Parameter search

As mentioned above, the `src/mff/agents/benchmark/AgentBenchmarkMetacentrum.java` class allows running the parameter search for `MFF A* Grid`, but the usage is not trivial.

Before attempting to run the full search, be warned that it requires almost 1000 days of CPU time and approximately 2 real-time days (if run on MetaCentrum and with enough computational nodes available). It also generates so many output files that opening the folder with them has the potential to crash Windows File Explorer.

If you want to rerun the experiment, run the `Metacentrum-Scripts/metascript-spec-grid.sh` script (uncomment the `qsub` line after testing it out) while having the `Metacentrum-Scripts/script-spec-grid.sh` script in the same folder. All of this should be done on the front node of some computational grid that supports the `qsub` command (e.g. MetaCentrum). The rest of the repository also needs to be present at a specific location, check (and update) the scripts if needed for this to match.

If you want to run only a part of the experiment locally, just run the `AgentBenchmarkMetacentrum` class as any other and input the command line parameters that you wish to test. Keep in mind that the first four parameters of the agent are input like this, the last one is set in code, so you might need to change that.

### Disclaimer

The results that you obtain while running experiments might differ from the ones stated in our works. The reason is that the agents' performance is influenced by the hardware capabilities that they are run on. To obtain interpretable results, make sure to run all agents that you compare on the same hardware, the relative results should stay the same.

### Copyrights
This framework is not endorsed by Nintendo and is only intended for research purposes. Mario is a Nintendo character which the authors don't own any rights to. Nintendo is also the sole owner of all the graphical assets in the game. Any use of this framework is expected to be on a non-commercial basis. The framework updates were created by David Šosvald as a bachelor and a master thesis at the Faculty of Mathematics and Physics of Charles University. The framework was created by [Ahmed Khalifa](https://scholar.google.com/citations?user=DRcyg5kAAAAJ&hl=en), based on the original Mario AI Framework by [Sergey Karakovskiy](https://scholar.google.se/citations?user=6cEAqn8AAAAJ&hl=en), [Noor Shaker](https://scholar.google.com/citations?user=OK9tw1AAAAAJ&hl=en), and [Julian Togelius](https://scholar.google.com/citations?user=lr4I9BwAAAAJ&hl=en), which in turn was based on [Infinite Mario Bros](https://fantendo.fandom.com/wiki/Infinite_Mario_Bros.) by Markus Persson.

### Contact

In case of any problems, questions, or suggestions, feel free to contact me at <sosvald@ksvi.mff.cuni.cz>.
