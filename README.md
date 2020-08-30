# BigData_SVM
- Predict whether income exceeds $50K/yr - SVM, Spark
- Extraction was done by Barry Becker from the 1994 Census database.

### 14 Statistic Features
|1|2|3|4|5|6|...|
|---------------------------------------------|---|---------|------|----------|-------------|---|
|less than 50K OR greater than or equal to 50K|Age|workclass|Weight|Eductation|Education-num|...|

### Preprocessed 123 features
- The first colmun indicates the class labels.

|1|2|3|4|5|...|124|
|---------------------------------------------|----|----|----|----------|---|--------------|
|less than 50K OR greater than or equal to 50K|Age1|Age2|Age3|workclass1|...|native-country|
