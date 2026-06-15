# **System Architecture Design Document: Active Hypothesis Refinement (AHR) Agent for ARC-AGI-3**

## **1\. System Overview & The Cold-Start Architecture**

To operate effectively within the hardware and time constraints of a Kaggle evaluation environment (12-hour compute limit, 0 internet access, local small-scale neural models), this system avoids semantic preconceptions. The agent is strictly un-opinionated upon initialization. It does not look for a "key" or a "door"; it looks for bounded pixel patterns and monitors numerical state streams.  
The core framework utilizes a Dual-Process model:

* **System 1 (Deterministic Physics & Graph-Search Engine):** Runs continuous, compiled spatial operations and expands candidate pathways using an active collection of programmatic rules.  
* **System 2 (Strategic Heuristic Planner):** A local, lightweight LLM (3B to 9B parameters) that wakes up exclusively when System 1 encounters an operational surprise, dead end, or environmental reset.

                     ┌──────────────────────────────┐  
                     │  Environment (64x64 Grid)     │  
                     └──────────────────────────────┘  
                        ▲                        │  
                Actions │                        │ State Frames (JSON)  
                        │                        ▼  
             ┌────────────────────┐     ┌───────────────────┐  
             │  System 1 Engine   │     │ Perception Layer  │  
             │ (Guarded DSL & BFS)│     │ (Flood-Fill, IDs) │  
             └────────────────────┘     └───────────────────┘  
                        ▲                        │  
         Updated Matrix │                        │ Forensic Trace  
         & Subgoals     │                        ▼  
                     ┌──────────────────────────────┐  
                     │  System 2 (Local LLM)        │  
                     │  High-Level Cost Architect   │  
                     └──────────────────────────────┘

## **2\. Epistemic Object Tracking (Managing the Unknown)**

Because the structural identity of objects is hidden until discovered through interaction, the perception layer handles spatial structures entirely separate from behavioral properties.

### **Phase A: Geometric Segmentation**

At step zero, a fast connected-component labeling algorithm (flood-fill) extracts every distinct component on the $64 \\times 64$ canvas. Each component is registered in memory using an immutable structure:

Python  
{  
    "Entity\_001": {  
        "spatial": {  
            "mask": SparseCoordinateArray,  \# Exact coordinate pairs  
            "bounding\_box": (min\_x, min\_y, max\_x, max\_y),  
            "size": Int,                    \# Total pixel count  
            "color": Int                    \# Palette integer (0-15)  
        },  
        "affordances": {  
            "is\_interactable": Unknown,  
            "is\_solid": Unknown,  
            "is\_controlled\_by\_player": Unknown  
        },  
        "dynamic\_state": {  
            "orientation\_index": 0,  
            "history\_deltas": \[\]  
        }  
    }  
}

### **Phase B: Unlabeled Affordance Discovery (The Probe Stage)**

Instead of attempting complex planning immediately, the agent defaults to a **Curiosity Probe Loop** to gather causal links:

1. **The Ego Check:** The engine fires the primary actions sequentially (Move UP, DOWN, LEFT, RIGHT). The frame delta engine registers which Entity\_ID coordinates move in direct synchronization with the action keys. That ID is permanently stamped: "is\_controlled\_by\_player": True.  
2. **The Structural Collisions:** The agent moves the player entity directly toward the nearest coordinate clusters.  
   * If a movement action is executed but the player coordinate stays static while a canvas delta indicates zero updates, the targeted cluster is tagged: "is\_solid": True (Wall/Boundary).  
   * If a collision occurs and a variable state changes elsewhere (e.g., another entity changes size, color, or steps through an internal sequence counter), both entities are linked inside a **Dynamic Relational Registry**.

## **3\. System 1: The Guarded Domain-Specific Language (DSL) & Simulation Engine**

To maintain search paths without running slow Python string iterations or model execution blocks during planning, all discovered mechanics are written into pre-compiled Python bytecode lambda pipelines.

### **Core DSL Primitives**

The execution framework relies on a flat collection of conditional evaluation blocks featuring two parts: a spatial or property validator (**Guard**) and a state modifier (**Effect**).

* **Relational Guards:**  
  * verify\_adjacency(Entity\_A, Entity\_B, orientation\_vector)  
  * verify\_overlap(Entity\_A, Entity\_B)  
  * evaluate\_property\_match(Entity\_A, attribute\_x, Entity\_B, attribute\_y)  
* **State Mutators (Effects):**  
  * transform\_position(Entity\_ID, offset\_vector)  
  * transform\_property(Entity\_ID, property\_string, operator, value)  
  * terminate\_state(type="WIN" | "LOSE")

### **Fast Transition Invalidation**

During Breadth-First Search (BFS) state expansions, the engine tracks every state permutation via a hash map of the raw canvas bytes. If an entry in the candidate rule group fails to accurately forecast a coordinate or attribute shift discovered during live tracking, it is pruned immediately from the system's operational graph model.

## **4\. System 2: Goal Orientation & Dynamic Cost Optimization**

When an environment features multi-step mechanics—such as scaling, sizing, or puzzle mechanisms where objects have identical colors but different scales—the goal matrices cannot be hardcoded. The local LLM tracks high-level properties and outputs structural cost vectors for the pathfinder rather than actions.

### **The Prompting Pipeline**

The prompt to the LLM ignores raw visual coordinate strings, preserving your tokens and remaining within the model context limit. It uses structural object descriptions:

Markdown  
\#\#\# SYSTEM STATE METADATA \#\#\#  
\- Active Entities:  
  \* Entity*\_001 (Player-Controlled): Size 4, Color 2, Location: Center*  
  *\* Entity\_*002 (Static Cluster): Size 9, Color 5, Configuration: Matrix Shape X  
  \* Entity*\_003 (Static Cluster): Size 16, Color 5, Configuration: Matrix Shape X*  
*\- Numerical State Variables:*  
  *\* Global\_*Var*\_0 (Tick Tracker): Decrements on every action step. Current value \= 35\.*

*\#\#\# RECENT HISTORICAL PROBE LOG \#\#\#*  
*\- Interaction: Player (Entity\_*001\) collided with Entity*\_002.*  
*\- Consequence: Entity\_*002 changed its orientation property to index 1\. Global*\_Var\_*0 decremented by 1\.  
\- Global Evaluation: No terminal victory triggered.

\#\#\# ANOMALY / SURPRISE REPORT \#\#\#  
The agent has navigated near Entity*\_003. Entity\_*002 and Entity*\_003 share identical structural shapes and color indices, but scale metrics differ (Size 9 vs Size 16).* 

*\#\#\# TASK \#\#\#*  
*Decompose the final objective and formulate the immediate Heuristic Evaluation Matrix Configuration for the local search algorithm. Optimize to prioritize property alignment over proximity.*

### **The Search Weight Structure**

The LLM reads this text summary and replies with a clean JSON block containing specific coefficients ($W$). These values direct the underlying pathfinding code:

JSON  
{  
  "active\_subgoal": "align\_structural\_dimensions",  
  "cost\_coefficients": {  
    "distance\_to\_entity\_002": \-15.0,  
    "distance\_to\_entity\_003": 0.0,  
    "global\_var\_0\_decrement\_cost": 45.0  
  },  
  "invalidation\_milestones": \[  
    "entity\_002.spatial.size \== entity\_003.spatial.size"  
  \]  
}

The System 1 pathfinder uses these coefficients to compute scores for prospective search directions on every step without requiring model execution:

$$\\text{State Evaluation Cost} \= (W\_{\\text{Entity\\\_002}} \\times D\_{\\text{Entity\\\_002}}) \+ (W\_{\\text{Var\\\_0}} \\times \\Delta\_{\\text{Var\\\_0}})$$  
This focuses the pathfinding engine entirely on interacting with Entity\_002 until the properties match the benchmark target Entity\_003, maintaining perfect action efficiency.

## **5\. Event-Driven Handoff Triggers**

To optimize execution speed and prevent timeout errors, the System 2 LLM remains idle. The System 1 search environment invokes the model only when it encounters specific programmatic exceptions:

1. **Epistemic Contradiction:** A live state transition breaks a current rule in the active candidate pool.  
2. **Resource Exhaustion Warning:** The planned steps required to reach a known target exceed the remaining resource limits (e.g., step limits or budget pools).  
3. **Milestone Achievement:** A condition listed within the JSON invalidation\_milestones evaluates to true, requiring the system to compute the next strategic phase.  
4. **Stagnation Timeout:** The agent performs 10 consecutive live actions without changing structural properties or shifting coordinates.