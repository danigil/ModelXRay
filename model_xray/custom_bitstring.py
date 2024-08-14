class Array_w_npslice(Array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deepcopy(self):
        return Array_w_npslice(copy.deepcopy(self.dtype), BitArray(copy.deepcopy(self.data.tobytes())))
        
    def shape(self):
        # print("Array_w_slice shape: (", len(self), self.itemsize, ")")
        return (len(self), self.itemsize)

    def _parse_slices(self, slice_0, slice_1):
        self_shape = self.shape()
        slice_0_new = [slice_0.start, slice_0.stop, slice_0.step]
        slice_1_new = [slice_1.start, slice_1.stop, slice_1.step]

        if slice_0.start is None:
            slice_0_new[0] = 0
        if slice_0.stop is None:
            slice_0_new[1] = self_shape[0]
        if slice_0.step is None:
            slice_0_new[2] = 1
    
        if slice_1.start is None:
            slice_1_new[0] = 0
        if slice_1.stop is None:
            slice_1_new[1] = self_shape[1]
        if slice_1.step is None:
            slice_1_new[2] = 1

        slice_0 = slice(*slice_0_new)
        slice_1 = slice(*slice_1_new)
        return slice_0, slice_1

    def __getitem__(self, key: Tuple[slice, slice]):
        slice_0, slice_1=self._parse_slices(*key)
        self_shape = self.shape()

        assert slice_0.stop - slice_0.start <= len(self)
        assert slice_1.stop - slice_1.start <= self.itemsize

        d = BitArray()
        for i in range(slice_0.start, slice_0.stop, slice_0.step):
            d.append(self.data[i*self_shape[1]+slice_1.start:i*self_shape[1]+slice_1.stop])

        ret = Array(f'bin{slice_1.stop-slice_1.start}')
        ret.data = d

        return ret

        # lindex = slice_0.start * self_shape[1]+slice_1.start
        # rindex = slice_0.stop * self_shape[1]+slice_1.stop

        # return Array(f'bin{slice_1.stop-slice_1.start}',self.data[lindex:rindex])

    def __setitem__(self, key: Tuple[slice, slice], value: Array):
        if 'bin' not in value.dtype.name:
            raise ValueError("Value must be a binary array")

        slice_0, slice_1=self._parse_slices(*key)
        self_shape = self.shape()

        if slice_0.stop - slice_0.start != len(value):
            raise ValueError("Shape dim0 mismatch")

        if slice_1.stop - slice_1.start != value.itemsize:
            raise ValueError("Shape dim1 mismatch")

        for i, offset in enumerate(range(slice_0.start, slice_0.stop)):
            # self.data[offset:offset+1:1] = f'0b{value[i]}'
            lindex = offset * self_shape[1]+slice_1.start
            # print("_setitem lindex: ", lindex)
            self.data.overwrite(f'0b{value[i]}',lindex)
