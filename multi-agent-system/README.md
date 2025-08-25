



### 1. Network Architecture
In a network architecture, every agent can communicate with every other agent. This creates a flexible, many-to-many communication pattern.
```mermaid
graph TD
    A[Agent 1] <--> B[Agent 2]
    A <--> C[Agent 3]
    B <--> C
    A <--> D[END]
    B <--> D
    C <--> D
```

### 2. Supervisor Architecture
In this pattern, a central supervisor coordinates all other agents, making routing decisions.
```mermaid
graph TD
    S[Supervisor] --> A1[Agent 1]
    S --> A2[Agent 2]
    A1 --> S
    A2 --> S
    S --> E[END]
```

### 3. Supervisor with Tool-Calling
This approach treats agents as tools that the supervisor can call, leveraging the LLM's natural tool-calling capabilities.
```mermaid
graph TD
    S[Supervisor LLM] --> T1[Tool: Agent 1]
    S --> T2[Tool: Agent 2]
    S --> T3[Tool: Agent 3]
    T1 --> S
    T2 --> S
    T3 --> S
```

### 4. Hierarchical Architecture
For complex systems, you can create hierarchies with supervisors managing teams of agents.
```mermaid
graph TD
    TS[Top Supervisor] --> T1[Team 1 Supervisor]
    TS --> T2[Team 2 Supervisor]
    T1 --> A1[Agent 1.1]
    T1 --> A2[Agent 1.2]
    T2 --> A3[Agent 2.1]
    T2 --> A4[Agent 2.2]
    A1 --> T1
    A2 --> T1
    A3 --> T2
    A4 --> T2
    T1 --> TS
    T2 --> TS
    TS --> E[END]
```

### 5. Custom Workflow Architecture
This approach defines explicit workflows with some dynamic routing capabilities.

```mermaid
graph TD
    A1[Agent 1] --> A2[Agent 2]
    A2 --> D{Decision}
    D -->|Condition A| A3[Agent 3]
    D -->|Condition B| A4[Agent 4]
    A3 --> A5[Agent 5]
    A4 --> A5
    A5 --> E[END]
```