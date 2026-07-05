# Editable CMTFlow Figure Legend

Files:
- `cmtflow_figure1_modules_editable.svg`: editable Figure 1 module architecture.
- `cmtflow_figure2_decision_flow_editable.svg`: editable Figure 2 daily decision flow.

PPT editing:
1. Insert the SVG into PowerPoint.
2. Right click the SVG and choose `Convert to Shape`.
3. Ungroup once or twice, then edit boxes, arrows, and text.
4. Keep the aspect ratio near 16:7 for paper-wide figures or crop to 16:9 for slides.

Color legend:
- Outer actor: blue `#3B82F6`, fill `#E8F1FF`.
- Controller: red `#E05A47`, fill `#FFF0EC`.
- Inner actor: green `#16A085`, fill `#EAF7F2`.
- Portfolio state/output: amber `#D59B18`, fill `#FFF7D7`.
- Data/state/context: gray `#B7C0CF`, fill `#F5F7FA`.
- Feedback/context arrows: dashed gray `#697386`.

Recommended PPT wording:
- Figure 1 title: `CMTFlow Decision Modules`.
- Figure 1 message: `Outer proposes what to hold; controller decides when to revise; inner refines daily weights.`
- Figure 2 title: `CMTFlow Daily Decision Flow`.
- Figure 2 message: `The controller compares the drifted current base with the outer candidate, then the inner actor refines the selected base.`

Notes:
- The controller should be shown as a learned daily event policy, not as an offline-label or frozen-stack post-processor.
- The inner actor should be labeled as `LSTM-attention local tilt`; avoid the obsolete convolutional-encoder wording.
- The final evaluation rule should be shown as `daily free decision, threshold 0.5, no hard min/max hold`.
