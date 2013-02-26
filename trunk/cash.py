#!/usr/bin/python

import sys
import os
import argparse
import logging
import logging.handlers

log = logging.getLogger("cash")

class ParseError(Exception):
    def __init__(self,message):
        Exception.__init__(self,message)

def raiseError(message,*args):
    msg = message % args
    raise ParseError("%s" % (msg,))

def raiseTokenError(token,message,*args):
    (kind,data,filename,line,col) = token
    msg = message % args
    raiseError("%s(%d,%d) %s" % (filename,line+1,col+1,msg))


def joinTokens(tokens):
    return "".join(map(lambda x:x[1],tokens))

def tokenAt(kind,data,tok):
    return (kind,data,tok[2],tok[3],tok[4])

class Lexer:
    def __init__(self):
        pass

    def charRange(start,stop):
        return map(chr,range(ord(start),ord(stop)+1))

    identStartChars = ['_','-'] + charRange('a','z') + charRange('A','Z')
    identChars = identStartChars + charRange('0','9') 
    lexChars = "{}():,$\"'\\"
    specialString = "\"\"\""
    newlineChars = "\n"
    spaceChars = " \t\v\r"

    def addText(self,tokens):
        if self.pos > self.textStart:
            tokens.append(("text",self.text[self.textStart:self.pos],self.filename,self.line,self.pos-self.linestart))
        else:
            self.textStart = self.pos

    def addToken(self,tokens,tok):
        self.addText(tokens)
        tokens.append((tok,tok,self.filename,self.line,self.pos-self.linestart))
        self.textStart = self.pos + len(tok)

    def addTokenSet(self,tokens,name,charset):
        self.addText(tokens)
        start = self.pos
        while self.text[self.pos+1] in charset and self.pos < len(self.text):
            self.pos = self.pos + 1
        tokens.append((name,self.text[start:self.pos+1],self.filename,self.line,start-self.linestart))
        self.textStart = self.pos+1

    def lex(self,filename):
        tokens = []
        self.pos = 0
        self.textStart = 0
        self.filename = filename
        self.line = 0
        self.linestart = 0

        with open(self.filename) as f:
            self.text = f.read()

        while self.pos < len(self.text):
            ch = self.text[self.pos]
            isText = False
            if ch in self.lexChars:
                if self.text[self.pos:].startswith(self.specialString):
                    self.addToken(tokens,self.specialString)
                else:
                    self.addToken(tokens,ch)
            if ch in self.newlineChars:
                self.addToken(tokens,ch)
                self.line += 1
                self.linestart = self.pos
            elif ch in self.identStartChars:
                self.addTokenSet(tokens,"ident",self.identChars)
            elif ch in self.spaceChars:
                self.addTokenSet(tokens,"space",self.spaceChars)
            self.pos = self.pos + 1
        return tokens

class Parser:
    def __init__(self,tokens=[]):
        self.tokens = tokens
        self.pos = 0

    def error(self,message,*args):
        if self.isEOF():
            self.pos -= 1 # back it up to the last token
        (tok,val,filename,line,col) = self.tokens[self.pos]
        msg = message % args
        raiseError("%s(%d,%d) %s" % (filename,line+1,col+1,msg))

    def isEOF(self):
        return self.pos >= len(self.tokens)

    def peek(self):
        return self.tokens[self.pos]

    def peekMatch(self,name,value=None):
        tok = self.peek()
        if tok[0] == name:
            if value != None:
                if tok[1] == value:
                    return True
                else:
                    return False
            return True
        else:
            return False

    def nextMatch(self,name,value=None):
        tok = self.peek()
        if tok[0] == name:
            if value != None:
                if tok[1] == value:
                    self.pos += 1
                    return True
                else:
                    return False
            self.pos += 1
            return True
        else:
            return False

    def nextMatchSequence(self,seq):
        idx = self.pos
        for tok in seq:
            if self.tokens[idx][0] != tok:
                return False
        self.pos += len(seq)
        return True

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def until(self,name):
        start = self.pos
        idx = self.pos
        while not self.peekMatch(name):
            self.next()
            if self.isEOF():
                raiseError("unexpected EOF, expected '" + name + "'")
        return self.tokens[start:self.pos]

    def through(self,name):
        value = self.until(name)
        self.next()
        return value

    def parseWhitespace(self):
        while not self.isEOF() and self.peekMatch("space"):
            self.next()

    def parseString(self):
        term = self.next()[0]
        start = self.pos
        while not self.isEOF():
            if self.peekMatch(term):
                self.next(len(term))
                return self.tokens[start:self.pos]
            elif self.peekMatch('\\'):
                self.next(2)
            else:
                self.next()
        raise ParseError("unexpected EOF, expected '" + term +"'")

    def matchStringStart(self):
        return self.peekMatch('"""') or self.peekMatch('"') or self.peekMatch("'")

    def parseThrough(self,term):
        # parse until next token(s), matching parens on the go
        start = self.pos
        level = 0
        while not self.isEOF():
            if self.peekMatch(term) and level == 0:
                result = self.tokens[start:self.pos]
                self.next()
                return result
            elif self.nextMatch('('):
                level = level+1
            elif self.nextMatch(')'):
                level = level-1
                if level < 0:
                    raiseError("unexpected ')'")
            elif self.matchStringStart():
                parseString()
            else:
                self.next()
        if level > 0:
            raiseError("unexpected EOF, expected closing ')'")
        else:
            raiseError("unexpected EOF, expected '" + term + "'")

    def parseArgument(self):
        start = self.pos
        while not self.isEOF():
            if self.peekMatch(','): break
            elif self.peekMatch(')'): break
            else: self.next()
        return self.tokens[start:self.pos]

    def parseIdentifier(self):
        if self.peekMatch("ident"):
            return self.next()[1]
        else:
            raiseError("expected identifier")

    def parseList(self,itemFn,delim,term):
        items = []

        self.parseWhitespace()
        if self.nextMatch(term):
            return items

        while not self.isEOF():
            self.parseWhitespace()
            items.append(itemFn())
            self.parseWhitespace()
            if self.nextMatch(term):
                break
            elif self.nextMatch(delim):
                continue
            else:
                raiseError("expected %s or %s",delim,term)
        return items

class Processor():
    def __init__(self):
        pass

    def buildMacroAST(self,parser):
        arguments = []
        name = parser.parseIdentifier()
        parser.parseWhitespace()
        if not parser.isEOF() and parser.nextMatch('('):
            for arg in parser.parseList(parser.parseArgument,',',')'):
                log.debug("MACRO-ARG",arg)
                arguments.append(self.buildAST(Parser(arg)))
        return (name,arguments)

    def buildAST(self,parser):
        (content,last) = self.buildNestedAST(parser)
        if last != None:
            parser.error("unexpected '%s'",last)
        return content

    def buildNestedAST(self,parser):
        output = []
        lastState = None

        while not parser.isEOF():
            startToken = parser.peek()
            if parser.nextMatch('$'):
                #log.debug(parser.peek())

                if parser.peekMatch('$'):
                    output.append(parser.next())

                elif parser.nextMatch('ident','define'):
                    log.debug("DEFINE")
                    parser.parseWhitespace()
                    ident = parser.parseIdentifier()
                    parser.parseWhitespace()

                    # handle optional arguments
                    params = []
                    if parser.nextMatch('('):
                        params = parser.parseList(parser.parseIdentifier,',',')')
                    parser.parseWhitespace()

                    # handle single/multi-line format
                    if parser.nextMatch(':'):
                        (content,last) = self.buildNestedAST(parser)
                        if last != "endef":
                            parser.error("expected endef")
                    else:
                        content = parser.until('\n')
                        parser.next()
                        content = self.buildAST(Parser(content))
                    data = (ident,params,content)
                    log.debug("DEFINED:",data)
                    output.append(tokenAt('define',data,startToken))

                elif parser.nextMatch('ident','endef'):
                    lastState = 'endef'
                    break

                elif parser.nextMatch('ident','include'):
                    parser.parseWhitespace()
                    if not parser.matchStringStart():
                        parser.error("expected string expression")
                    filename = parser.parseString()
                    nested = Parser(filename,Lexer().lex(filename))

                elif parser.nextMatch('ident','if'):
                    parser.parseWhitespace()
                    ifStatement = self.buildAST(Parser(parser.through(':')))
                    (trueBranch,last) = self.buildNestedAST(parser)
                    root = tokenAt('if',[ifStatement,trueBranch,[]],startToken)
                    tok = root

                    # handle optional elif statements
                    while last == 'elif':
                        parser.parseWhitespace()
                        ifStatement = self.buildAST(Parser(parser.through(':')))
                        startToken = parser.peek()
                        (trueBranch,last) = self.buildNestedAST(parser)
                        nextToken = tokenAt('if',[ifStatement,trueBranch,[]],tok)

                        tok[1][2].append(nextToken)
                        tok = nextToken

                    # handle optional else statement
                    if last == 'else':
                        if not parser.nextMatch(':'):
                            parser.error("expected ':'")
                        (falseBranch,last) = self.buildNestedAST(parser)
                        tok[1][2] = falseBranch

                    # must terminate with an endif
                    if last != 'endif':
                        parser.error("expected endif") 

                    log.debug("IF-DUMP",root)

                    output.append(root)

                elif parser.nextMatch('ident','elif'):
                    lastState = 'elif'
                    break

                elif parser.nextMatch('ident','else'):
                    lastState = 'else'
                    break

                elif parser.nextMatch('ident','endif'):
                    lastState = 'endif'
                    break

                elif parser.nextMatch('{'):
                    content = self.buildAST(Parser(parser.through('}')))
                    output.append(tokenAt('eval',content,startToken))

                elif parser.nextMatch('('):
                    data = self.buildMacroAST(parser)
                    output.append(tokenAt('macro',data,startToken))
                    parser.parseWhitespace()
                    if not parser.nextMatch(')'):
                        parser.error("extra chars in macro expression")

                else:
                    data = self.buildMacroAST(parser)
                    output.append(tokenAt('macro',data,startToken))
            else:
                output.append(parser.next())

        return output,lastState

    def evaluateAST(self,parser,defines={},wrapEval=False):
        output = []

        while not parser.isEOF():
            tok = parser.peek()
            data = tok[1]
            log.debug(tok)

            if parser.nextMatch('macro'):
                # find the macro
                (name,arguments) = data
                if name not in defines: 
                    parser.error("macro %s is not defined",name)
                (params,content) = defines[name]

                # set default params and arguments
                context = defines.copy()
                for param in params:
                    context[param] = ([],None)
                for idx in range(min(len(params),len(arguments))):
                    argContent = arguments[idx]
                    argContent = self.evaluateAST(Parser(argContent),defines)
                    context[params[idx]] = ([],arguments[idx])

                result = self.evaluateAST(Parser(content),context)
                log.debug("EVAL-RESULT: %s",result)
                if wrapEval:
                    data = joinTokens(result).replace("\\","\\\\")
                    data = '"""' + data + '"""'
                    result = [tokenAt('text',data,tok)]

                output += result

            elif parser.nextMatch('define'):
                #TODO: warn about macro redefinition? 
                (name,params,content) = data
                defines[name] = (params,content)

            elif parser.nextMatch('if'):
                (ifStatement,trueBranch,falseBranch) = data

                def isDefined(name): return name in defines

                # evaluate the statement to wrap any $macro statements
                content = self.evaluateAST(Parser(ifStatement),defines,True)
                expr = joinTokens(content)
                try:
                    result = eval(
                        "bool(%s)"%(expr,),{
                        "defined": isDefined
                    })
                except Exception,e:
                    raiseTokenError(tok,"cannot evaluate if expression - %s",str(e))

                # branch based on eval result
                if result:
                    output += self.evaluateAST(Parser(trueBranch),defines)
                else:
                    log.debug("FALSE-BRANCH %s",falseBranch)
                    output += self.evaluateAST(Parser(falseBranch),defines)

            elif parser.nextMatch('eval'):
                log.debug("EVAL: %s",data)
                expr = joinTokens(self.evaluateAST(Parser(data),defines,True))
                try:
                    log.debug("EVAL: str(%s)",expr)
                    result = eval("str(%s)"%(expr,))
                except Exception,e:
                    raiseTokenError(tok,"cannot evaluate eval expression - %s",str(e))
                output.append(tokenAt('text',result,tok))

            else:
                output.append(parser.next())

        return output

    def process(self,filename,defines):
        # lexical pass
        output = Lexer().lex(filename)

        # parse pass
        parser = Parser(output)
        output = self.buildAST(parser)

        # sanity check
        if not parser.isEOF():
            log.debug("TOKENS:",parser.peek())
            raiseEerror("Some tokens were not consumed by processing.")

        # evaluate
        parser = Parser(output)
        output = self.evaluateAST(parser,defines)

        # turn all output tokens back into text
        return joinTokens(output)

def main():
    defines = {}


    # process arguments
    parser = argparse.ArgumentParser(
        description="Text file preprocessor"
    )
    parser.add_argument("filename",help="File to process by Cash",type=str)
    parser.add_argument("-E","--environ",action="store_true",help="Define all environment vars as macros")
    #parser.add_argument("-D","--define",action="append",help="Defines a macro")
    parser.add_argument("-d","--debug",action="store_true",help="Show debug output")
    parser.add_argument("-o","--output",type=str,default=None,help="Output to file")
    args = parser.parse_args()

    # bring in environment vars
    if args.environ:
        for name in os.environ:
            defines[name] = ([],[('text',os.environ[name],None,0,0)])
    
    # handle inline defines 
    #for userDefine in args.define:

    # setup logging
    rootLogger = logging.getLogger()
    rootLogger.addHandler(logging.StreamHandler(sys.stdout)) 
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(handler)
    log.propagate = False

    # set log level
    if args.debug:
        log.setLevel(logging.DEBUG) 
    else:
        log.setLevel(logging.WARNING) 

    p = Processor()
    try:
        result = p.process(args.filename,defines)
        if args.output is None:
            print result
        else:
            with open(args.output,"wt+") as f:
                f.write(result)
    except ParseError,e:
        log.error(str(e))

if __name__ == '__main__':
    main()
