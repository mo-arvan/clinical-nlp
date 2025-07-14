# Cervical Cancer

## Existing Issues and Limitations

- Overlapping labels causes issues. For example, "HGSIL, high grade dysplasia CIN II-III" matches CIN II, however, it should match CIN II-III. 
- Text referring to historical information should be ignored. 
- Behavior in export is not entirely clear. What should happen when there are more than one instance of a certain label?


## Development Loop

Start with a simple baseline, then: 

- Perform error analysis, identify issues and limitations. 
- Update the model
- Note selection strategy? 
- Define alerts and warnings in case performance drops on previously annotated data.
- Reviewing process: 
  - Identify notes with the most errors
  - Identify notes that contain labels with the worst performance
- Final Report, Visualization, Human-Computer Interaction
- Continuous evaluation and monitoring, performance indicators and alerts 
- Expanding to other clinics and cancer types
- Quantifying the impact of the system
- Fairness and bias

- report level to patient level
- how to find matching patients
- age?

- Reviewing Guide Video
- Error Review (two columns, error classification, additional comments, reviewer name)
- Late August 

New data:
- weekly export 

https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/cervixca.html

Saraiya M, Colbert J, Bhat GL, Almonte R, Winters DW, Sebastian S, O'Hanlon M, Meadows G, Nosal MR, Richards TB, Michaels M, Townsend JS, Miller JW, Perkins RB, Sawaya GF, Wentzensen N, White MC, Richardson LC. Computable Guidelines and Clinical Decision Support for Cervical Cancer Screening and Management to Improve Outcomes and Health Equity. J Womens Health (Larchmt). 2022 Apr;31(4):462-468. doi: 10.1089/jwh.2022.0100. PMID: 35467443; PMCID: PMC9206487.