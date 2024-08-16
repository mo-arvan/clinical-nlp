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


New data:
- weekly export 