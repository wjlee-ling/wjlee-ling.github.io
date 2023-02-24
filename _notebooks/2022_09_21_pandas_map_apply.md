# Pandas Map, Apply ApplyMap
> map, apply, applymap 함수 차이점

- toc: true
- badges: true
- comments: true
- categories: [TIL]

Pandas를 다루다 보면 한번쯤은 헷갈릴만한 map, apply, applymap 정리하기.


```python
#hide
import pandas as pd
```

# Map

*Map values of Series according to an input mapping or function. Used for substituting each value in a Series with another value, that may be derived from a function, a dict or a Series.* (출처: pandas [doc](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html))

**Series** 대상으로만 사용 가능하며, 주로 series내 값들을 다른 값들로 바꿔주는 데 사용한다.


```python
df = pd.DataFrame({'name':['James', 'Hugh', 'Laurie'], 'age':[32, 18, 27], 'sex': ['male', 'male', 'female']})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>




```python
# DataFrame에 적용 불가능
try:
    df.map({'female': 0, 'male':1})
    df
except Exception as e:
    print(e)
```

    'DataFrame' object has no attribute 'map'


매핑할 pair들은 dictionary꼴로 넣어주거나 함수, 또는 또다른 series로 넣어줄 수 있다. 


```python
# 1. dictionary
df['sex_converted'] = df['sex'].map({'female':0, 'male':1})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
      <th>sex_converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2. function
df['sex_converted'] = df['sex'].map(lambda x: 1 if x=='male' else 0)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
      <th>sex_converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3. series
mapping_series = pd.Series({'male':1, 'female':0})
df['sex_converted'] = df['sex'].map(mapping_series)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
      <th>sex_converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Apply

series에만 적용가능한 map과 달리, apply는 series/dataframe에 사용가능한 메소드이다. 

apply는 map과 달리 지정해 주는 axis에 따라 주어진 함수를 적용한다. (axis=0이면 column, axis=1이면 row 대상) 따라서 map과 달리 apply는 지정된 axis를 따라 function를 적용할 수 있다.


```python
df = pd.DataFrame({'name':['James', 'Hugh', 'Laurie'], 'age':[32, 18, 27], 'sex': ['male', 'male', 'female']})
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df = df.apply( lambda x: x.sum() if x.dtype == 'int64' else x, axis=0)
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>77</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>77</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>77</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



# Applymap

*Apply a function to a Dataframe elementwise. This method applies a function that accepts and returns a scalar to every element of a DataFrame.*

applymap은 **dataframe**에서만 사용가능한 메소드이다. applymap은 apply와 달리 모든 element에 같은 함수를 적용하기 때문에 axis를 지정할 필요가 없다.


```python
df = pd.DataFrame({'name':['James', 'Hugh', 'Laurie'], 'age':[32, 18, 27], 'sex': ['male', 'male', 'female']})
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>32</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hugh</td>
      <td>18</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laurie</td>
      <td>27</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.applymap(lambda x:len(str(x)))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


