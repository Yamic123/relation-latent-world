# Relational Latent World Learning

## A Proposal for Effective Variable Discovery

## Research Background and Core Problem

### Can an agent discover the effective variables of the world by itself?

Current artificial intelligence systems have achieved remarkable
progress in learning representations, predicting future states, and
controlling complex environments. However, a fundamental question
remains unresolved:

Can an agent autonomously discover the effective variables that govern
the world, rather than relying on predefined representations or
human-designed concepts?

This research does not aim to discover a single universal representation
that explains all phenomena across all domains and scales. Instead, it
investigates how an intelligent agent can construct effective latent
structures within a specific domain, scale, and abstraction level.

------------------------------------------------------------------------

# I. Research Gap

## Existing world models learn latent states, but not latent structure

Most existing world models focus on learning latent states useful for
prediction or control.

Representative directions include: - predictive latent representation
learning; - object-centric representation learning; - latent dynamics
modeling; - causal representation learning.

However, several questions remain:

1.  Effective variables are usually assumed rather than discovered.

2.  Latent representations may achieve prediction accuracy without
    semantic identifiability.

3.  Relations among variables are underexplored.

Existing methods often focus on:

\[ Observation ightarrow Latent State \]

while understanding the world requires:

\[ Variables + Relations \]

4.  Task-specific causal structures are usually predefined rather than
    generated from shared latent structures.

------------------------------------------------------------------------

# II. World View and Physical / Structural Assumptions

## 1. Latent worlds are domain-, scale-, and level-specific

We do not assume a universal latent world that explains all aspects of
reality.

Different domains, abstraction levels, and scales may require different
latent worlds.

A latent world is a specialized structure for modeling a particular
aspect of reality.

------------------------------------------------------------------------

## 2. Base latent world as an abstract knowledge substrate

For a specific domain, scale, and abstraction level:

\[ W_D=(Z_D,R_D) \]

where:

-   (Z_D): effective latent variables;
-   (R_D): relational operators among variables.

The base latent world is not the fundamental components of physical
reality. It is an abstraction enabling prediction, reasoning, planning,
intervention, and control.

------------------------------------------------------------------------

## 3. Variables possess intrinsic identity and relational structure

Variable identity is not completely determined by relationships.

Variables correspond to real regularities of the world.

However, relational structures provide additional constraints for
identifiability:

\[ Identity = Intrinsic Representation + Relational Position \]

------------------------------------------------------------------------

## 4. Relation operators define latent world structure

Variables interact through relational mechanisms such as:

-   enhancement;
-   inhibition;
-   balance;
-   transformation.

Relations are represented as learnable operators:

\[ R\_{ij}:z_iightarrow z_j \]

------------------------------------------------------------------------

## 5. Task transformations generate causal hierarchies

Task-specific causal structures emerge through transformation:

\[ G_T=`\phi`{=tex}\_T(W_D) \]

Different tasks reorganize the same latent substrate into different
causal structures.

Example:

Object manipulation latent world:

\[ (object,pose,force,contact) \]

Grasping:

\[ object pose ightarrow grasp point ightarrow contact force ightarrow
lifting \]

Pushing:

\[ object position ightarrow contact direction ightarrow friction
ightarrow trajectory \]

------------------------------------------------------------------------

# III. Relational Latent World

The core representation is:

\[ W_D=(Z_D,R_D) \]

where latent variables and relational operators are jointly learned.

------------------------------------------------------------------------

# IV. Research Questions

1.  Can relation operators enhance latent variable identifiability?
2.  How can stable relations be learned from video and interaction data?
3.  How can task transformations be learned?
4.  How can cross-task generalization be achieved within one latent
    world?

------------------------------------------------------------------------

# V. Research Roadmap

## Phase 1

Validate relation-aware latent models in controlled environments.

## Phase 2

Test task generalization in robotics manipulation.

## Phase 3

Study latent transformation and causal structure reorganization.

------------------------------------------------------------------------

# Core Hypothesis

Intelligence is not about finding a single fixed world representation,
but about constructing relational latent worlds across different
domains.
