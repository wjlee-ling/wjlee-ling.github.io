---
title: Data Structure Study (1) Array
description: Arrays
search: true
toc: true
date: 2023-02-26
categories:
    - data structure
tags:
    - c++
    - array
---

체계적으로 알고리즘 공부를 해 보기로 마음먹었다. 우선 가장 기본이 되는 데이터 구조부터 정리하기로 했다. 프로그래머스와 백준 허브로 코딩 테스트 대비 데이터 구조 관련 문제들을 풀어왔긴 했는데, 복잡도와 로직을 정리할 겸 LeetCode의 [Data Structures and Algorithms](https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/) 코스를 결제했다. 아직 수업 초반부라 그런지 주로 쓰는 언어인 Python으로는 쉽게 풀 수 있는 문제들이 많아 이번 기회에 C++를 배워 문제를 풀어보기로 했다. LeetCode는 문제 위주로 알고리즘을 설명하기 때문에 이론을 보충하고자 [MIT 6.006 Introduction to Algorithms, Spring 2020](https://www.youtube.com/playlist?list=PLUl4u3cNGP63EdVPNLG3ToM6LaEUuStEY) 내용도 같이 정리해 본다.

## Array

프로그래밍 언어마다 "array"의 정의는 다를 수 있지만 python과 c++에서 array는 **contiguous**한 데이터 구조를 말한다. 물론 python에서 array에 해당되는 데이터 구조는 **list**로, list는 엄밀히 말하면 **dynamic array**이다. **Static array**는 한번 선언하면 그 크기를 바꿀 수 없는 반면 **dynamic array**는 크기 변경이 가능하다. 이에 두 데이터 구조의 작업(operation) 복잡도가 다른 경우가 있다.

### Complexity

|           **operation**           	| **Static Array** 	| **Dynamic Array** 	| **String** 	|
|:---------------------------------:	|:----------------:	|:-----------------:	|:----------:	|
| random access or modification at  i-th|     O(1)       	|        O(1)       	|    O(1)       |
| insertion/deletion, not from end  	|         _        	|        O(n)       	|    O(n)    	|
|    insertion/deletion, from end   	|         _        	|        O(1)       	|    O(n)    	|
|             scaling up            	|         -        	|   O(1) amortized  	|    O(n)    	|

i번째 요소에 접근하거나 변경하기 위해서는 해당 요소의 주소를 알아야 하는데 이는 array의 주소(==첫번째 요소의 주소)에다가 i의 배수(C++에서 `sizeof(int)`)를 더하면 되기 때문에(∴ contiguous) 상수만큼의 시간이 걸린다.
크기를 변경할 수 있는 dynamic array의 경우 요소를 추가하거나 제거할 때 해당 요소의 위치에 따라 복잡도가 달라진다. 맨 마지막이 아닌 위치에 새 요소를 추가하거나 해당 위치의 요소를 제거할 때는 요소가 추가/제거된 새로운 array를 처음부터 다시 만들어야 하기 때문에 O(n)의 복잡도를 갖는다. 요소의 개수가 dynamic array의 크기보다 클 때에는 더 큰 메모리 공간으로 옮겨야 하는데, (python의 list처럼) 2의 배수로 크기를 키운다고 한다면 (1)에 의해 *평균적으로* O(1)의 복잡도를 갖게 된다.

$$ 
\begin{equation}
\theta(1+2+4+8+16+32+\cdots+n) = \theta(\sum_{i=1}^{\log{n}}{2^i}) \\
= \theta(2^{\log{n}+1} - 1) = \theta(n)
\end{equation}
$$

## C++ Array 정의
**reference**
1. [UW: C++ arrays](http://courses.washington.edu/css342/zander/css332/array.html)
2. [UW: Pointers](http://courses.washington.edu/css342/zander/css332/pointers.html)
3. [geeksforgeeks: Arrays in C++](https://www.geeksforgeeks.org/arrays-in-c-cpp/?ref=lbp)
4. [Programiz: C++ Memory Management](https://www.programiz.com/cpp-programming/memory-management)

### static array 정의
C++에서 static array는 컴파일링할 때 메모리 할당이 된다.
```cpp
int arr1[5]; // not initialized
int arr2[5] {1, 2, 3, 4, 5}; // initialized
```
pointer를 이용하면 runtime에 메모리를 할당하는 array를 만들 수 있다.
```cpp
int* arr = new int[3]; // declare pointer and dynamically allocate memory 
*arr = {1, 2, 3}; // assign value
delete arr; // deallocate the memory
```
C++에서 `*`는 정의하는 변수가 (reference가 아닌) pointer임을 의미하고 `new`는 메모리를 할당한다는 뜻이다. 즉 `arr3`라는 포인터와 `int[3]`의 array를 연결한다.

> ❗️ c++에서는 pointer가 아닌 reference로 array를 만들 수 없다. 예를 들어 ```cpp int& arr[3] {1,2,3}```을 하면 ``` 'arr' declared as array of references of type 'int &'```라는 에러가 발생한다.

### dynamic array 정의
LeetCode에서는 Standard Library에 있는 `vector`을 적극적으로 활용한다. 
```cpp
#include <vector>
using namespace std;
vector<int> arr3;
arr3.push_back(1);
arr3.push_back(2);
```

## two pointers 
2개의 포인트로 string과 array를 순회하는 알고리즘.

index `0`과 `array.length-1`을 값으로 갖는 변수 `left`와 `right`가 `while left < right`를 만족할 때 일련의 작업을 하는 식으로 구현할 수 있다. while-loop을 돌지만 최대 문장의 길이(`array.legnth`)만큼만 돌기 때문에 평균 시간 복잡도는 O(n)이다. 문제에 따라 `left`와 `right`가 각기 다른 arr/str을 순회하거나 `right`가 `left`와 같이 인덱스 0에서 시작하게 만들 수도 있다.

### pseudo code
```python
function fn(arr):
    left = 0, right = arr.length-1;
    while left < right:
        task-specific operations
        left += 1 
        right -= 1
```

### 문제
* [167. Two Sum II - Input Array Is Sorted](https://github.com/wjlee-ling/algorithms/tree/main/0167-two-sum-ii-input-array-is-sorted)
* [reverse string](https://github.com/wjlee-ling/algorithms/tree/main/reverse-string)
* [Reverse Only Letters](https://github.com/wjlee-ling/algorithms/tree/main/0917-reverse-only-letters)
* [392. Is Subsequence](https://github.com/wjlee-ling/algorithms/tree/main/0392-is-subsequence)

## sliding window
window라는 sub-array를 만들어 주어진 array 위를 순회하는 알고리즘.

two pointers와 비슷하게 `left`와 `right`라는 인덱스를 만들지만, 이 인덱스들은 window의 시작/끝 인덱스이며 내부 로직도 two pointers와 조금 다르다.

### pseudo code
```python
function fn(arr):
    left = 0
    for right in [0, arr.length-1]:
        logic to add arr[right] to window

        while (left < right) & (condition not met):
            logic to remove arr[left] from window
            left += 1
```
for-loop과 while-loop이 있기 때문에 $$O(n^2)$$ 의 시간 복잡도를 갖는다고 생각할 수 있는데 $$O(n)$$ 이다. while-loop이 매 `right`값마다 `arr.length`만큼 돌 필요가 없고, `right`가 최대 `arr.length`만큼 돌 때 `left`도 최대 `arr.length`만큼 돌기 때문에 amortized O(n)의 시간 복잡도를 갖게 된다.

### 문제
* [713. Subarray Product Less Than K](https://github.com/wjlee-ling/algorithms/tree/main/0713-subarray-product-less-than-k)
* [Maximum Average Subarray I](https://github.com/wjlee-ling/algorithms/tree/main/maximum-average-subarray-i)
* [Max Consecutive Ones III](https://github.com/wjlee-ling/algorithms/tree/main/0max-consecutive-ones-iii)

## prefix sum
누적 합계 array를 만들어 문제를 해결하는 알고리즘.

`arr`가 주어질 때, `prefix[i] = prefix[i-1] + arr[i]`이다. 이 `prefix`라는 새 array를 만들 때 O(n)의 시간이 걸리지만, 이후 연산은 (복잡한 작업이 필요 없을 시) O(1)로 가능하다. 별도로 새 array 없이 in-place로 가능한 경우도 있다.

### 문제
* [724. Find Pivot Index](https://github.com/wjlee-ling/algorithms/tree/main/0724-find-pivot-index)
* [1208. Get Equal Substrings Within Budget](https://github.com/wjlee-ling/algorithms/tree/main/1208-get-equal-substrings-within-budget)
* [303. Range Sum Query - Immutable](https://github.com/wjlee-ling/algorithms/tree/main/0303-range-sum-query-immutable)