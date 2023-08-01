def hello(n: list[str]) -> None:
    for name in n:
        print('Hello ' + name)

n_1 = ['Joao', 'Luis' , 'Fe']

n_2 = [1, 2, 3]

hello(n_1) #Ok

hello(n_2) #-----> Mypy error message: Argument 1 to "hello" has incompatible type "List[int]"; expected "List[str]"