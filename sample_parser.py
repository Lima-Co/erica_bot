
#-*- coding: utf-8 -*-

if __name__ == "__main__":
    with open('sample.txt', 'r', encoding='UTF8') as f:
        req_lines = []
        rep_lines = []
        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if i % 2 == 0:
                req_lines.append(line)
            else:
                rep_lines.append(line)
            print( i % 2)
