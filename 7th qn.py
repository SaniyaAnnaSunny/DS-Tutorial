import numpy as np
from itertools import combinations

class MarketBasketAnalysis:
    def __init__(self, min_supp=0.5, min_conf=0.7):
        """
        Initialize parameters for mining frequent itemsets and generating association rules.
        
        Parameters:
        - min_supp: Minimum threshold for item support (float between 0 and 1)
        - min_conf: Minimum threshold for rule confidence (float between 0 and 1)
        """
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.freq_patterns = None
        self.rules = None
    
    def _compute_support(self, item_grp, dataset):
        """Calculate the fraction of transactions containing the given item group."""
        count = sum(1 for record in dataset if item_grp.issubset(record))
        return count / len(dataset)
    
    def _generate_combinations(self, itemsets, size):
        """Generate new candidate itemsets of a given size."""
        items = sorted(set(element for subset in itemsets for element in subset))
        return list(combinations(items, size))
    
    def _filter_combinations(self, candidate_sets, prev_patterns, size):
        """Prune candidate sets that do not have all their subsets in previous frequent patterns."""
        return [candidate for candidate in candidate_sets if all(frozenset(subset) in prev_patterns for subset in combinations(candidate, size-1))]
    
    def analyze(self, dataset):
        """
        Identify frequent itemsets and derive association rules.
        
        Parameters:
        - dataset: List of transactions (each transaction is a list of items)
        """
        dataset = [frozenset(transaction) for transaction in dataset]  # Convert to frozensets
        
        self.freq_patterns = {}
        size = 1
        
        # Identify frequently occurring single items
        elements = {item for record in dataset for item in record}
        candidates = [frozenset([item]) for item in elements]
        
        while candidates:
            # Evaluate support of each candidate set
            valid_candidates = [(candidate, self._compute_support(candidate, dataset)) for candidate in candidates]
            valid_candidates = [(candidate, support) for candidate, support in valid_candidates if support >= self.min_supp]
            
            if valid_candidates:
                self.freq_patterns[size] = valid_candidates
                size += 1
                
                # Generate next-level candidates
                candidates = self._generate_combinations([itemset for itemset, _ in valid_candidates], size)
                
                # Apply pruning step
                prev_patterns = {itemset for itemset, _ in valid_candidates}
                candidates = self._filter_combinations(candidates, prev_patterns, size-1)
            else:
                break
        
        # Generate association rules
        self.rules = []
        for size, patterns in self.freq_patterns.items():
            if size == 1:
                continue
            
            for itemset, support in patterns:
                for i in range(1, size):
                    for lhs in combinations(itemset, i):
                        lhs = frozenset(lhs)
                        rhs = itemset - lhs
                        
                        # Compute confidence of the rule
                        lhs_support = self._compute_support(lhs, dataset)
                        confidence = support / lhs_support
                        
                        if confidence >= self.min_conf:
                            self.rules.append({
                                'lhs': lhs,
                                'rhs': rhs,
                                'support': support,
                                'confidence': confidence
                            })
    
    def get_frequent_patterns(self, size=None):
        """
        Retrieve frequent itemsets of a specified size.
        
        Parameters:
        - size: The size of itemsets to retrieve (None for all sizes)
        
        Returns:
        - List of frequent itemsets with support values
        """
        if size is None:
            return [pattern for level in self.freq_patterns.values() for pattern in level]
        return self.freq_patterns.get(size, [])
    
    def get_association_rules(self):
        """Retrieve association rules that meet the minimum confidence threshold."""
        return self.rules

# Example execution
if __name__ == "__main__":
    # Sample dataset representing customer purchases
    dataset = [
        ['bread', 'milk'],
        ['bread', 'diapers', 'beer', 'eggs'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['bread', 'milk', 'diapers', 'beer'],
        ['bread', 'milk', 'diapers', 'cola']
    ]
    
    # Instantiate and execute analysis
    mba = MarketBasketAnalysis(min_supp=0.4, min_conf=0.6)
    mba.analyze(dataset)
    
    # Display frequent itemsets
    print("Frequent Itemsets:")
    for size, patterns in mba.freq_patterns.items():
        print(f"\nSize {size}:")
        for pattern, support in patterns:
            print(f"{set(pattern)}: support = {support:.2f}")
    
    # Display association rules
    print("\nAssociation Rules:")
    for rule in mba.get_association_rules():
        print(f"{set(rule['lhs'])} => {set(rule['rhs'])} "
              f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f})")
