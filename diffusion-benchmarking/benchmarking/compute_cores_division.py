

def integer_triples(X: int):
    triples = []
    for a in range(1, X + 1):
        if X % a != 0:
            continue
        for b in range(1, X + 1):
            if (X // a) % b != 0:
                continue
            c = X // (a * b)

            triples.append([a, b, c])
    return triples

def integer_doubles(X: int):
    doubles = []
    for a in range(1, X + 1):
        if X % a != 0:
            continue
        b = X // a
        doubles.append([a, b])
    return doubles


if __name__ == "__main__":
    X = int(input("Enter a positive integer X: "))
    result = integer_doubles(X)
    print(f"All positive integer doubles (a, b) such that a*b = {X}:")
    print(", ".join(str([1] + double) for double in result))