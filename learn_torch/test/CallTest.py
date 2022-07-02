class Person:
    def __call__(self, name):   # 重载()运算符
        print("__call__" + "hello_" + name)

    def hello(self, name):
        print("hello" + name)


person = Person()
person("zhangsan")
person.hello("lisi")
