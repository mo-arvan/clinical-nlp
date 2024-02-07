# Dataset Info

Downloaded from: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Dataset

2010 - Relations

- 2010 i2b2/VA Challenge on Concepts, Assertions, and Relations in Clinical Text
    - Training Data: Concept assertion relation training data
    - Test Data: Reference standard for test data
    - Test Data: Test data

## Annotation Format

Source: dataset docs, 'Annotation File Formatting.pdf'

`c=”concept text” offset||t=”concept type”||a=”assertion value”`

where 
- c: c represents a mention of a concept. concept text is replaced with the actual text from the report.
- offset: offset represents the beginning and end line and word numbers that span the concept text. An offset is
  formatted as the line number followed by a colon followed by the word number. The starting and ending offset are
  separated by a space. The first line of a report starts as line number 1. The first word in a line is counted as word
  number 0.
- t: t represents the semantic type of concept mentioned. concept type is replaced with problem, treatment, or test
- a: a represents the assertion of the concept mentioned. assertion value is
  replaced with present, absent, possible, conditional, hypothetical, or
  associated with someone else.

Examples:

```text
c=”prostate cancer” 5:7 5:8||t=”problem”||a=”present”
c=”diabetes” 2:14 2:14||t=”problem”||a=”absent”
c=”pain” 7:3 7:3||t=”problem”||a=”conditional”
```

## Issues

### 1) Inconsistent Annotation Format

The offsets are numbers, line number is 1 indexed, word number is 0 indexed; this inconsistency may cause confusion in
parsing the data.

In order to convert them to indices, we need to add one to word number end, and subtract one from line number start.

### 2) Incorrect Annotation

In file `270045381.ast`, line #60:

`c="few bacteria" 159:31 159:33||t="problem"||a="present"`

The line:
```python
line = "The patient urinalysis on 10/19/97 showed moderate occult blood , pH 5 , albumin 1+ , white blood cells present , 2-50 red blood cell , 10-20 white blood cells , few bacteria , and moderate bladder epithelial cells ."
```
> 

Instead of 31:33 (`line.split()[31:34]`), it should be 31:32 (`line.split()[31:33]`). Refer to conversion process
described in `1) Inconsistent Annotation Format`.

In file `145980160.ast`, line #61:

`c="non-insulin diabetes mellitus" 30:2 30:5||t="problem"||a="present"`

The line:
```python
line = "Significant for non-insulin diabetes mellitus , for which he takes Diabeta , one QD ; right eye cataract , operated on three years ago ."
```
Instead of 2:5 (`line.split()[2:6]`), it should be 2:4 (`line.split()[2:5]`). Refer to conversion process described in `1)


### 3) Duplicate Annotations

From file `641557794_WGH.ast`, line 2 and 23:
```text
c="papillary carcinoma" 50:0 50:1||t="problem"||a="present"
c="papillary carcinoma" 50:0 50:1||t="problem"||a="present"
```

## Examples

Present: problems associated with the patient can be present. This is the
default category for medical problems and it contains that do not fit the
definition of any of the other assertion category.
*the wound* was noted to be clean with mild serous drainage
history of *chest pain*
patient had *a stroke*
the patient experienced a the drop in hematocrit
the patient has had increasing weight gain
He has pneumonia
2) Absent: the note asserts that the problem does not exist in the patient. This
category also includes mentions where it is stated that the patient HAD a
problem, but no longer does.
patient denies pain
no fever




no history of diabetes
No pneumonia was suspected
History inconsistent with stroke
his dyspnea resolved
elevated enzymes resolved
3) Possible: the note asserts that the patient may have a problem, but there is
uncertainty expressed in the note. Possible takes precedence over absent, so
terms like “probably not” or “unlikely” categorize problems as being possible
just as “probably” and “likely” do.
This is very likely to be an asthma exacerbation .
Doctors suspect an infection of the lungs .
The patient came in to rule out pneumonia .
Questionable / small chance of pneumonia .
Pneumonia is possible / probable
Suspicion of pneumonia
We are unable to determine whether she has leukemia.
It is possible / likely / thought / unlikely that she has pneumonia
We suspect this is not pneumonia
this is probably not cancer
pneumonia unlikely
4) Conditional: the mention of the medical problem asserts that the patient
experiences the problem only under certain conditions. Allergies can fall into
this category.
Patient has had increasing dyspnea on exertion
Penicillin causes a rash
Patient reports shortness of breath upon climbing stairs.
5) Hypothetical: medical problems that the note asserts the patient may
develop.
If you experience wheezing or shortness of breath
Ativan 0.25 to 0.5 mg IV q 4 to 6 hours prn anxiety
6) Not associated with Patient: the mention of the medical problem is associated
with someone who is not the patient.
Family history of prostate cancer
Brother had asthma