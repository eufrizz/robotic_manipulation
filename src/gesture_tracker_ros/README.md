## Medipipe handlandmarker
Uses around 120% CPU and can't keep up at 20Hz and eventually freezes. 15Hz OK but high latency, probably will eventually give out. Publish rate hovers around 9Hz.
Should profile to see where the latency/processing goes, but I'm guessing it's in inference.
Solutions:
- C++ - dunno if this would actually help if inference is the bottleneck
- Run on GPU - seems like it will require compiling Mediapipe from source with Tensorflow GPU support - PITA. https://chuoling.github.io/mediapipe/getting_started/gpu_support.html
