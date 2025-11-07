```mermaid
graph TD
    A[Input] --> B[Agent/Nbr轨迹编码]
    A --> C[地图编码]
    B --> D[Reprogramming层]
    C --> D
    D --> E[LLM融合]
    E --> F[Decoder输出]
    F --> G[动力学约束]
```