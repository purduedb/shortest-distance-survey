# Source: https://github.com/Jiboxiake/HCL-Python/blob/main/HCL.ipynb
import struct
from typing import List, Optional
from typing import List, Optional

class Partition:
    def __init__(self, partition_bitvector: int = 0):
        """
        Represents a Partition with a bitvector.

        :param partition_bitvector: Encoded partition bitvector.
        """

def count_trailing_zeros_64(x: int) -> int:
    x = x & 0xFFFFFFFFFFFFFFFF  # force to 64 bits
    if x == 0:
        return 64
    return (x & -x).bit_length() - 1

class FlatCutIndex:
    def __init__(self, partition_bitvector: int = 0, dist_index: Optional[List[int]] = None, distances: Optional[List[int]] = None):
        """
        Represents a FlatCutIndex.

        :param partition_bitvector: Encoded partition bitvector.
        :param dist_index: List of distance indices.
        :param distances: List of distances.
        """
        self.partition_bitvector = partition_bitvector
        self.dist_index = dist_index if dist_index is not None else []
        self.distances = distances if distances is not None else []

    def partition(self) -> int:
        """
        Extracts the partition from the partition bitvector.

        :return: Partition value.
        """
        return self.partition_bitvector >> 6

    def cut_level(self) -> int:
        """
        Extracts the cut level from the partition bitvector.

        :return: Cut level value.
        """
        return self.partition_bitvector & 63

    def size(self) -> int:
        """
        Calculates the size of the FlatCutIndex in bytes.

        :return: Size in bytes.
        """
        return len(self.distances) * 4 + len(self.dist_index) * 2 + 8  # 8 bytes for partition_bitvector

    def label_count(self) -> int:
        """
        Returns the number of labels stored.

        :return: Label count.
        """
        return len(self.distances)

    def cut_size(self, cl: int) -> int:
        """
        Returns the number of labels at a given cut level.

        :param cl: Cut level.
        :return: Number of labels at the given cut level.
        """
        if cl == 0:
            return self.dist_index[0]
        return self.dist_index[cl] - self.dist_index[cl - 1]

    def bottom_cut_size(self) -> int:
        """
        Returns the number of labels at the lowest cut level.

        :return: Number of labels at the lowest cut level.
        """
        return self.cut_size(self.cut_level())

    def empty(self) -> bool:
        """
        Checks if the FlatCutIndex is empty.

        :return: True if empty, False otherwise.
        """
        return len(self.distances) == 0 and len(self.dist_index) == 0 and self.partition_bitvector == 0

    def cl_begin(self, cl: int) -> List[int]:
        """
        Returns the start of distance labels for a given cut level.

        :param cl: Cut level.
        :return: List of distances starting at the given cut level.
        """
        offset = self.dist_index[cl - 1] if cl > 0 else 0
        return self.distances[offset:]

    def cl_end(self, cl: int) -> List[int]:
        """
        Returns the end of distance labels for a given cut level.

        :param cl: Cut level.
        :return: List of distances ending at the given cut level.
        """
        return self.distances[:self.dist_index[cl]]

    def unflatten(self) -> List[List[int]]:
        """
        Returns labels in a list-of-lists format.

        :return: List of lists of distances.
        """
        labels = []
        for cl in range(self.cut_level() + 1):
            labels.append(self.distances[self.dist_index[cl - 1] if cl > 0 else 0:self.dist_index[cl]])
        return labels

    def __repr__(self):
        """
        String representation of the FlatCutIndex.

        :return: String representation.
        """
        return f"FlatCutIndex(partition_bitvector={self.partition_bitvector}, dist_index={self.dist_index}, distances={self.distances})"

    def is_same(self, other: 'FlatCutIndex') -> bool:
        """
        Checks if two FlatCutIndex instances are the same.

        :param other: Another FlatCutIndex instance.
        :return: True if they are the same, False otherwise.
        """
        return (self.partition_bitvector == other.partition_bitvector and
                self.dist_index == other.dist_index and
                self.distances == other.distances)


    def get_lca(self, other: 'FlatCutIndex') -> int:
       cut_level_s = self.cut_level()
       cut_level_o = other.cut_level()
       lca_level = min(cut_level_s, cut_level_o)
       p1,p2 = self.partition(), other.partition()
       if p1!=p2:
           p3 = p1^p2
           diff_level = count_trailing_zeros_64(p3)
           if diff_level < lca_level:
               lca_level = diff_level

       return lca_level

class ContractionLabel:
    def __init__(self, cut_index: Optional[FlatCutIndex] = None, distance_offset: int = 0, parent: int = None):
        """
        Represents a contraction label.

        :param cut_index: Instance of FlatCutIndex or equivalent data structure.
        :param distance_offset: Distance to the node owning the labels (default is 0).
        :param parent: Parent node in the tree rooted at the label-owning node (default is None).
        """
        self.cut_index = cut_index if cut_index is not None else FlatCutIndex()
        self.distance_offset = distance_offset
        self.parent = parent

    def size(self) -> int:
        """
        Calculates the size of the contraction label in bytes.

        :return: Size of the contraction label.
        """
        total_size = self.__sizeof__()
        if self.distance_offset == 0 and self.cut_index is not None:
            total_size += self.cut_index.size()
        return total_size

    def __repr__(self):
        """
        String representation of the ContractionLabel.

        :return: String representation.
        """
        return f"ContractionLabel(cut_index={self.cut_index}, distance_offset={self.distance_offset}, parent={self.parent})"

class ContractionIndex:
    def __init__(self, labels: List[ContractionLabel]):
        """
        Represents a contraction index.

        :param labels: List of ContractionLabel objects.
        """
        self.labels = labels
        self.merge_map = {}
        for i in range(1, len(labels)):
            entry = labels[i]
            if entry.parent!=None and entry.parent != 0:
                parent=entry.parent
                while entry.parent !=0 and entry.parent != None:
                    parent = entry.parent
                    entry = labels[entry.parent]
                self.merge_map[i]=parent



    def get_distance(self, v: int, w: int) -> int:
        """
        Computes the distance between two nodes using the contraction index.

        :param v: Node ID of the first node.
        :param w: Node ID of the second node.
        :return: Distance between the two nodes.
        """
        cv = self.labels[v]
        cw = self.labels[w]
        #print(cv)
        #print(cw)
        #assert not cv.cut_index.empty() and not cw.cut_index.empty()
        same_flag = False
        if cv.cut_index.empty() and cw.cut_index.empty():
                p1 = self.merge_map.get(v)
                p2 = self.merge_map.get(w)
                cvv = self.labels[p1]
                cww = self.labels[p2]
                assert not cvv.cut_index.empty() and not cww.cut_index.empty()
                if cvv.cut_index.is_same(cww.cut_index):
                    same_flag=True
        elif cv.cut_index.is_same(cw.cut_index):
            same_flag=True
        else:
            same_flag=False

        if same_flag:
            if v == w:
                return 0
            if cv.distance_offset == 0:
                return cw.distance_offset
            if cw.distance_offset == 0:
                return cv.distance_offset
            if cv.parent == w:
                return cv.distance_offset - cw.distance_offset
            if cw.parent == v:
                return cw.distance_offset - cv.distance_offset

             # Find the lowest common ancestor
            v_anc, w_anc = v, w
            cv_anc, cw_anc = cv, cw
            while v_anc != w_anc:
                if cv_anc.distance_offset < cw_anc.distance_offset:
                    w_anc = cw_anc.parent
                    cw_anc = self.labels[w_anc]
                elif cv_anc.distance_offset > cw_anc.distance_offset:
                    v_anc = cv_anc.parent
                    cv_anc = self.labels[v_anc]
                else:
                    v_anc = cv_anc.parent
                    w_anc = cw_anc.parent
                    cv_anc = self.labels[v_anc]
                    cw_anc = self.labels[w_anc]

            return cv.distance_offset + cw.distance_offset - 2 * cv_anc.distance_offset
        """
        if cv.cut_index.is_same(cw.cut_index):
            if cv.cut_index.empty() and cw.cut_index.empty():
                p1 = self.merge_map.get(v)
                p2 = self.merge_map.get(w)
                cvv = self.labels[p1]
                cww = self.labels[p2]
                assert not cvv.cut_index.empty() and not cww.cut_index.empty()
                if cvv.cut_index.is_same(cww.cut_index):
                    if v == w:
                        return 0
                    if cv.distance_offset == 0:
                        return cw.distance_offset
                    if cw.distance_offset == 0:
                        return cv.distance_offset
                    if cv.parent == w:
                        return cv.distance_offset - cw.distance_offset
                    if cw.parent == v:
                        return cw.distance_offset - cv.distance_offset
                    # Find the lowest common ancestor
                    v_anc, w_anc = v, w
                    cv_anc, cw_anc = cv, cw
                    while v_anc != w_anc:
                        if cv_anc.distance_offset < cw_anc.distance_offset:
                            w_anc = cw_anc.parent
                            cw_anc = self.labels[w_anc]
                        elif cv_anc.distance_offset > cw_anc.distance_offset:
                            v_anc = cv_anc.parent
                            cv_anc = self.labels[v_anc]
                        else:
                            v_anc = cv_anc.parent
                            w_anc = cw_anc.parent
                            cv_anc = self.labels[v_anc]
                            cw_anc = self.labels[w_anc]
                    return cv.distance_offset + cw.distance_offset - 2 * cv_anc.distance_offset

            else:#actually the same cut index
                if v == w:
                    return 0
                if cv.distance_offset == 0:
                    return cw.distance_offset
                if cw.distance_offset == 0:
                    return cv.distance_offset
                if cv.parent == w:
                    return cv.distance_offset - cw.distance_offset
                if cw.parent == v:
                    return cw.distance_offset - cv.distance_offset

                # Find the lowest common ancestor
                v_anc, w_anc = v, w
                cv_anc, cw_anc = cv, cw
                while v_anc != w_anc:
                    if cv_anc.distance_offset < cw_anc.distance_offset:
                        w_anc = cw_anc.parent
                        cw_anc = self.labels[w_anc]
                    elif cv_anc.distance_offset > cw_anc.distance_offset:
                        v_anc = cv_anc.parent
                        cv_anc = self.labels[v_anc]
                    else:
                        v_anc = cv_anc.parent
                        w_anc = cw_anc.parent
                        cv_anc = self.labels[v_anc]
                        cw_anc = self.labels[w_anc]

                return cv.distance_offset + cw.distance_offset - 2 * cv_anc.distance_offset
        """

        # Fallback to hierarchical distance computation
        result =  cv.distance_offset + cw.distance_offset
        if cv.cut_index.empty():
            assert(cv.parent is not None and cv.parent != 0)
            cv = self.labels[self.merge_map.get(v)]
        if cw.cut_index.empty():
            assert(cw.parent is not None and cw.parent != 0)
            cw = self.labels[self.merge_map.get(w)]
        return result+self.get_hierarchical_distance(cv.cut_index, cw.cut_index) #cv.distance_offset + cw.distance_offset + self.get_hierarchical_distance(cv.cut_index, cw.cut_index)

    def get_hierarchical_distance(self, a: FlatCutIndex, b: FlatCutIndex) -> int:
        """
        Computes the hierarchical distance between two FlatCutIndex objects.

        :param a: First FlatCutIndex.
        :param b: Second FlatCutIndex.
        :return: Hierarchical distance.
        """
        #cut_level = min(a.cut_level(), b.cut_level())#Libin: double check the cut level implementation
        cut_level = a.get_lca(b)
        a_offset = a.dist_index[cut_level - 1] if cut_level > 0 else 0
        b_offset = b.dist_index[cut_level - 1] if cut_level > 0 else 0
        #print(cut_level)
        a_end = min((a.dist_index[cut_level]-a_offset), (b.dist_index[cut_level]-b_offset))

        min_dist = float('inf')
        for i in range(0,a_end):
            dist = a.distances[a_offset+i] + b.distances[b_offset+i]
            if dist < min_dist:
                min_dist = dist

        """
        for i in range(a_offset, a_end):
            dist = a.distances[i] + b.distances[i]
            if dist < min_dist:
                min_dist = dist
        """

        return min_dist

class HCL:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.parse_file()

    def parse_file(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            print(line)
