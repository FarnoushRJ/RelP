<center>
    <img src='assets/RelP_logo.jpeg', width='300'>
    <h1>Faithful and Efficient Circuit Discovery via Relevance Patching</h1>
</center>

## 🔖 Overview
<p align="justify">Activation patching is a standard method in mechanistic interpretability for localizing the components of a model responsible for specific behaviors, but it is computationally expensive to apply at scale. Attribution patching offers a faster, gradient-based approximation, yet suffers from noise and reduced reliability in deep, highly non-linear networks. In this work, we introduce Relevance Patching (RelP), which replaces the local gradients in attribution patching with propagation coefficients derived from Layer-wise Relevance Propagation (LRP). LRP propagates the network's output backward through the layers, redistributing relevance to lower-level components according to local propagation rules that ensure properties such as relevance conservation or improved signal-to-noise ratio. Like attribution patching, RelP requires only two forward passes and one backward pass, maintaining computational efficiency while improving faithfulness. We validate RelP across a range of models and tasks, showing that it more accurately approximates activation patching than standard attribution patching, particularly when analyzing residual stream and MLP outputs in the Indirect Object Identification (IOI) task. For instance, for MLP outputs in GPT-2 Large, attribution patching achieves a Pearson correlation of 0.006, whereas RelP reaches 0.956, highlighting the improvement offered by RelP. Additionally, we compare the faithfulness of sparse feature circuits identified by RelP and Integrated Gradients (IG), showing that RelP achieves comparable faithfulness without the extra computational cost associated with IG.</p>

## 📜 Propagation Rules for Transformers
The following rules are currently implemented and supported.

| **Layer**            | **Propagation Rule** |
|-----------------------|----------------------|
| LayerNorm / RMSNorm   | LN-rule [(Ali et al., 2022)](https://proceedings.mlr.press/v162/ali22a.html) |
| Activation Function (e.g., GELU)           | Identity-rule [(Jafari et al., 2024)](https://neurips.cc/virtual/2024/poster/96794) |
| Linear                | 0-rule [(Montavon et al., 2019)](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf) |
| Attention             | AH-rule [(Ali et al., 2022)](https://proceedings.mlr.press/v162/ali22a.html) |
| Multiplicative Gate   | Half-rule [(Jafari et al., 2024)](https://neurips.cc/virtual/2024/poster/96794); [(Arras et al., 2019)](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_11) |

---

## 🙏 Acknowledgment
We build on the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/tree/main) library, which provides the foundation for much of this code.

