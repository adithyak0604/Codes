class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables=variables
        self.domains=domains
        self.constraints=constraints
        self.assignments={var:None for var in variables}
    
    def is_consistent(self, var, value):
        for constraint in self.constraints[var]:
            other_var, relation=constraint
            if self.assignments[other_var] is not None:
                if relation=='!=' and value==self.assignments[other_var]:
                    return False
                elif relation=='==' and value!=self.assignments[other_var]:
                    return False
        return True    
    def backtrack(self):
        for var in self.variables:
            if self.assignments[var] is None:
                for value in self.domains[var]:
                    if self.is_consistent(var, value):
                        self.assignments[var]=value
                        if self.backtrack():
                            return True
                        self.assignments[var]=None
                return False
        return True

    def solve(self):
        if self.backtrack():
            return self.assignments
        else:
            return None
    
variables=['A','B','C','D']
domains={ 'A':[1,2,3,4],'B':[1,2,3,4,],'C':[1,2,3,4,],'D':[1,2,3,4,] }
constraints={ 'A':[('B','!='),('C','!='),('D','!=')],
              'B':[('A','!='),('C','!='),('D','!=')],
              'C':[('A','!='),('B','!='),('D','!=')],
              'D':[('A','!='),('B','!='),('C','!=')]  }
csp=CSP(variables, domains, constraints)
solution=csp.solve()
print(solution)