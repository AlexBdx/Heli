Overview:
Better results with this run. The f1_score passed 0.2 for the first time!
The residual connection block is functional and seems to improve recall notably.

Things to update:
- A lot of the best runs have used the extreme param range. Re-run with larger boundaries.
- Still stuck at 110 fps, no clear path on higher FPS yet
- Implement a different stabilization algo using phase correlation.
- Capture a new video with the pan/tilt head instead

Best:
gaussWindow 	mgp 	minArea 	residualConnections 	winSize 	maxLevel 	diffMethod 	realFps 	avNbBoxes 	avNbFilteredBoxes 	avNbHeliBox 	percentHeliTotalFiltered 	percentFrameWithHeli 	f1_score
31 	9 	150 	16 	2 	3 	7 	0 	99.746619 	5.901532 	0.927061 	0.519329 	0.560189 	0.383552 	0.22767
58 	11 	175 	25 	4 	3 	7 	0 	99.373959 	6.643535 	0.870709 	0.461651 	0.530201 	0.43377 	0.238581
No stabilization:
3 	125 	1 	3 	3 	7 	0 	411.845572 	2.280726 	2.177374 	0.794693 	0.364978 	0.791377 	0.249781
26 	3 	125 	1 	5 	3 	7 	25 	0 	337.947404 	2.137255 	2.044818 	0.85014 	0.415753 	0.844228 	0.278568
150 	3 	125 	1 	4 	3 	7 	25 	0 	3 	301.215293 	1.745455 	1.658741 	0.861538 	0.519393 	0.856745 	0.32336
16 	3 	125 	1 	3 	3 	7 	25 	0 	5 	317.629731 	1.472067 	1.407821 	0.863128 	0.613095 	0.859527 	0.357846
With stab:



