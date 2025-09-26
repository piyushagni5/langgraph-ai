### Building a Self-Improvement Agent with LangGraph: Reflection Vs Reflexion

Imagine an AI that doesn’t just generate answers but also reviews its own outputs, finds weak spots, and improves the next draft, i.e. actually **learns from its own mistakes**. That’s what reflection agents are built for: self-correction, iterative refinement, and strategy improvement over time.

> Reflection is a prompting strategy  used to improve the quality and success rate of agents and similar AI systems.
> 

Think of it as a prompting strategy that asks an LLM to reflect and critique its previous actions. Sometimes the reflector also uses external knowledge, like tool outputs or retrievals, to make the critique more accurate. The result is not a single-shot response but a short feedback loop where generation and review alternate until the output quality improves.

[Reflection](https://blog.langchain.com/reflection-agents/) systems usually fall into three broad categories:

- **Basic Reflection Agent**: A simple generator plus reflector loop. The generator drafts content, the reflector critiques it, and the generator revises. This is lightweight and effective for many editing tasks.
- **Reflexion Agent**: A more structured approach that tracks past actions, hypotheses, and reflections in a traceable log. Reflexion is useful for problem solving where the agent needs to learn from multiple failed attempts.
- **Language Agent Tree Search (LATS)**: A search-style strategy that explores multiple action branches, reflects on outcomes, and prunes or keeps promising branches. Best for planning and multi-step reasoning.

In this post, we’ll focus on Reflection and Reflexion agents, explore their workflows, and implement them step-by-step using LangChain and LangGraph.


### Understanding Basic Reflection Agent

A **Reflection Agent** is an AI system that goes beyond just generating answers, it actively critiques its own outputs and keeps refining them. You can think of it as a loop between two roles:

- **Generator**: drafts the initial response.
- **Reflector**: reviews that draft, points out flaws or gaps, and suggests improvements.

This back-and-forth runs for a few iterations, and with each cycle, the output gets more polished, reliable, and useful. The beauty here is that the AI essentially learns from its own mistakes in real time, almost like a writer rewriting their draft after a round of editorial feedback.

![](https://cdn-images-1.medium.com/max/1600/0*q1EAlBH2ZjHg08jl.png)

Credits: [LangChain](https://blog.langchain.com/reflection-agents/)

In this section, we’ll build a Reflection Agent for a LinkedIn post generator using LangGraph, a framework designed for creating self-improving AI systems. The idea is to design a workflow that mimics human-like reflective thinking, where the agent doesn’t just stop at the first draft, but keeps refining until the content feels polished and engaging.

By the end of this walkthrough, you’ll see how to set up both the generator and the reflector roles, use LangChain for structured prompting, and leverage LangGraph to stitch everything together into an iterative feedback loop.

Let’s start with our first implementation for the **basic Reflection pattern** through a LinkedIn content creation agent. The flow is simple yet powerful: the agent drafts a post, a separate “reflector” persona critiques it, and then the system revises the content based on that feedback.


### Understanding Reflexion Agent

The [**Reflexion pattern**](https://arxiv.org/pdf/2303.11366), introduced by Shinn et al., extends basic reflection by combining self-critique with external knowledge integration and structured output parsing.

Unlike simple reflection, Reflexion allows an agent to learn from mistakes in real time while leveraging additional information. 

The workflow typically follows these steps:

1. **Initial Generation:** The agent produces a response along with self-critique and research queries.
2. **External Research:** Knowledge gaps identified during critique trigger web searches or other information retrieval.
3. **Knowledge Integration:** New insights are incorporated into an improved response.
4. **Iterative Refinement:** The agent repeats the cycle until the response meets desired quality thresholds.

![Credits: LangChain](https://cdn-images-1.medium.com/max/1600/0*VibsYc06HxrsDUXo.png)

Credits: [LangChain](https://blog.langchain.com/reflection-agents/)

In a [Reflexion agent](https://arxiv.org/pdf/2303.11366), the system is structured into three interconnected roles: **Actor, Evaluator, and Self-Reflection**.

- The **Actor** attempts the task: writing code, solving a problem, or taking actions in an environment.
- The **Evaluator** provides internal feedback, assessing the quality of the Actor’s output.
- The **Self-Reflection** module generates textual reflections, which capture what went wrong or could be improved.

These reflections are stored in **memory**:

- **Short-term memory** tracks the trajectory of the current attempt.
- **Long-term memory** accumulates lessons from prior reflections, guiding future iterations.

![](https://cdn-images-1.medium.com/max/1600/1*InX9g39nPoYnRtjGTbjxiQ.png)

(a) Diagram of Reflexion. (b) Reflexion reinforcement algorithm (Credits: [Shinn et al., 2023](https://arxiv.org/pdf/2303.11366))

The process is iterative: the Actor tries, the Evaluator scores, the Self-Reflection critiques, and the Actor leverages that feedback for the next attempt. This loop continues until the task succeeds or a maximum number of iterations is reached.

For example, if the Actor fails a step, the reflection might note:

> “I got stuck in a loop; try a different strategy or tool next time.”
> 

The next iteration, informed by this reflection, is more likely to succeed. Reflexion can handle diverse feedback, numeric rewards, error messages, or human hints, all integrated into the reflection process.

Reflexion has demonstrated remarkable results. On coding benchmarks like **HumanEval**, a Reflexion-augmented GPT-4 agent reached **91% success**, compared to 80% without reflection. In decision-making simulations (AlfWorld), ReAct + Reflexion agents solved **130 out of 134 challenges**, clearly outperforming non-reflective counterparts.

This highlights the core power of Reflexion: by structuring AI agents to **think about their actions and retain lessons learned**, they continuously improve, tackling complex tasks more effectively over time.


## Prerequisites

Before setting up this repository, ensure you have the following installed:

- Python 3.12 or higher
- UV package manager (recommended) or pip
- Git

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
cd langgraph-ai
```

### Step 2: Install UV Package Manager

If you haven't installed UV yet, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Create Virtual Environment

Navigate to the specific project directory you want to work with. For example, to work with the Adaptive RAG system:

```bash
cd langgraph-cookbook/agentic-patterns
```

Create a virtual environment using UV:

```bash
uv venv --python 3.12.4
```

### Step 4: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

**Using UV (Recommended):**
```bash
uv pip install -r requirements.txt
```

**Using pip (Alternative):**
```bash
pip install -r requirements.txt
```

### Step 6: Adding Virtual Environment to Jupyter Kernel
To use your UV virtual environment with Jupyter notebooks, you need to install ipykernel and register the environment as a kernel:
**Install ipykernel in the virtual environment**:
   ```bash
   uv pip install ipykernel
   ```

**Register the virtual environment as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=langgraph-ai --display-name="LangGraph AI"
   ```
When you open a notebook, you can select the "LangGraph AI" kernel from the kernel menu.

### Step 7: Environment Configuration

Create a `.env` file in your project directory with the necessary API keys:

```env
ANTHROPIC_API_KEY="your-anthropic-api-key"
# LANGCHAIN_API_KEY="your-langchain-api-key"  # optional
# LANGCHAIN_TRACING_V2=True                   # optional
# LANGCHAIN_PROJECT="multi-agent-swarm"       # optional
```

**Note**: The `LANGCHAIN_API_KEY` is required if you enable tracing with `LANGCHAIN_TRACING_V2=true`.

## License

This project is open source and available under the MIT License.

## Acknowledgements

- Original LangChain repository:
    - [Reflection](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection)
    - [Reflexion](https://github.com/langchain-ai/langgraph/tree/main/docs/docs/tutorials/reflexion)

- Reflexion: Language Agents with Verbal Reinforcement Learning by [Shinn](https://arxiv.org/pdf/2303.11366) et al., 2023

