#!/usr/bin/python

class MacroDef(str):
    def __init__(self,*args):
        str.__init__(self,*args)
    def __str__(self,*args):
        return self(*args)

class __foobar(MacroDef):
    def __call__(self,*args):
        return "1"
foobar = __foobar()
print foobar
print foobar()
print int(foobar)
