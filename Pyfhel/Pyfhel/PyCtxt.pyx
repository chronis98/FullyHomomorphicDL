# distutils: language = c++
# cython: language_level=3, boundscheck=False
"""PyCtxt. Ciphertext of Pyfhel, Python For Homomorphic Encryption Libraries.
"""
# -------------------------------- IMPORTS ------------------------------------
# Import Pyfhel and PyPtxt for operations
from .Pyfhel import Pyfhel
from .PyPtxt import PyPtxt

# Encoding types: 0-UNDEFINED, 1-INTEGER, 2-FRACTIONAL, 3-BATCH
from Pyfhel.util import ENCODING_t

# Dereferencing pointers in Cython in a secure way
from cython.operator cimport dereference as deref

# ----------------------------- IMPLEMENTATION --------------------------------
cdef class PyCtxt:
    """Ciphertext class of Pyfhel, contains a value/vector of encrypted ints/doubles.

    This class references SEAL, PALISADE and HElib ciphertexts, using the one 
    corresponding to the backend selected in Pyfhel. By default, it is SEAL.   
    """
    def __cinit__(self,
                  PyCtxt copy_ctxt=None,
                  Pyfhel pyfhel=None,
                  fileName=None,
                  serialized=None,
                  encoding=None):
        if (copy_ctxt): # If there is a PyCtxt to copy, override all arguments and copy
            self._ptr_ctxt = new Ciphertext(deref(copy_ctxt._ptr_ctxt))
            self._encoding = copy_ctxt._encoding
            if (copy_ctxt._pyfhel):
                self._pyfhel = copy_ctxt._pyfhel
        
        else:
            self._ptr_ctxt = new Ciphertext()
            if fileName:
                if not encoding:
                    raise TypeError("<Pyfhel ERROR> PyCtxt initialization with fileName requires valid encoding")    
                self.from_file(fileName, encoding)
            elif serialized:
                if not encoding:
                    raise TypeError("<Pyfhel ERROR> PyCtxt initialization from serialized requires valid encoding")    
                self.from_bytes(serialized, encoding)
            else:
                self._encoding = to_ENCODING_t(encoding) if encoding else ENCODING_T.UNDEFINED
            if (pyfhel):
                self._pyfhel = pyfhel
            
    def __dealloc__(self):
        if self._ptr_ctxt != NULL:
            del self._ptr_ctxt
    
    def __init__(self,
                  PyCtxt copy_ctxt=None,
                  Pyfhel pyfhel=None,
                  fileName=None,
                  serialized=None,
                  encoding=None):
        """__init__(PyCtxt copy_ctxt=None, Pyfhel pyfhel=None, fileName=None, serialized=None, encoding=None)

        Initializes an empty PyCtxt ciphertext.
        
        To fill the ciphertext during initialization you can:
            - Provide a PyCtxt to deep copy. 
            - Provide a pyfhel instance to act as its backend.
            - Provide a fileName and an encoding to load the data from a saved file.

        Attributes:
            copy_ctxt (PyCtxt, optional): Other PyCtxt to deep copy.
            pyfhel (Pyfhel, optional): Pyfhel instance needed to operate.
            fileName (str, pathlib.Path, optional): Load PyCtxt from this file.
                            Requires non-empty encoding.
            serialized (bytes, optional): Read PyCtxt from a bytes serialized string, 
                            obtained by calling the to_bytes method.
            encoding (str, type, int, optional): encoding type of the new PyCtxt.
        """
        pass

    @property
    def _encoding(self):
        """ENCODING_t: returns the encoding type.
        
        Can be set to: 0-UNDEFINED, 1-INTEGER, 2-FRACTIONAL, 3-BATCH

        See Also:
            :func:`~Pyfhel.util.to_ENCODING_t`

        :meta public:
        """
        return ENCODING_t(self._encoding)
    
    @_encoding.setter
    def _encoding(self, new_encoding):
        if not isinstance(new_encoding, ENCODING_t):
            raise TypeError("<Pyfhel ERROR> Encoding type of PyPtxt must be ENCODING_t")        
        self._encoding = new_encoding.value
        
    @_encoding.deleter
    def _encoding(self):
        self._encoding = ENCODING_t.UNDEFINED.value
        
    @property
    def _pyfhel(self):
        """A Pyfhel instance, used for operations"""
        return self._pyfhel
    @_pyfhel.setter
    def _pyfhel(self, new_pyfhel):
        if not isinstance(new_pyfhel, Pyfhel):
            raise TypeError("<Pyfhel ERROR> new_pyfhel needs to be a Pyfhel class object")       
        self._pyfhel = new_pyfhel 
        
    cpdef int size_capacity(self) except +:
        """Maximum size the ciphertext can hold.
        
        Return:
            int: allocated size for this ciphertext
        """
        return self._ptr_ctxt.size_capacity()
     
    cpdef int size(self) except +:
        """Current size of the ciphertext.
        
        Return:
            int: size of this ciphertext"""
        return self._ptr_ctxt.size()

    @property    
    def capacity(self):
        """int: Maximum size the ciphertext can hold."""
        return self._ptr_ctxt.size_capacity()

    @property
    def size(self):
        """int: Actual size of the ciphertext."""
        return self._ptr_ctxt.size()

    @property
    def noiseBudget(self):
        """int: Noise budget.
        
        A value of 0 means that it cannot be decrypted correctly anymore.
        
        See Also:
            :func:`~Pyfhel.Pyfhel.noiseLevel`
        """
        return self._pyfhel.noiseLevel(self)

    # =========================================================================
    # ================================== I/O ==================================
    # =========================================================================
    def __reduce__(self):
        """__reduce__()

        Required for pickling purposes. Returns a tuple with:
            - A callable object that will be called to create the initial version of the object.
            - A tuple of arguments for the callable object.
        """
        return (PyCtxt, (None, None, None, self.to_bytes(), self._encoding))

    cpdef void to_file(self, fileName) except +:
        """to_file(Path fileName)
        
        Alias of `save` with input sanitizing.

        Args:
            fileName: (str, pathlib.Path) File where the ciphertext will be stored.

        Return:
            None
        """
        self.save(_to_valid_file_str(fileName))

    cpdef void save(self, str fileName) except +:
        """save(str fileName)
        
        Save the ciphertext into a file. The file can new one or
        exist already, in which case it will be overwriten.

        Args:
            fileName: (str) File where the ciphertext will be stored.

        Return:
            None            
        """
        cdef ofstream* outputter
        cdef string bFileName = fileName.encode('utf8')
        outputter = new ofstream(bFileName, binary)
        try:
            self._ptr_ctxt.save(deref(outputter))
        finally:
            del outputter

    cpdef bytes to_bytes(self) except +:
        """to_bytes()

        Serialize the ciphertext into a binary/bytes string.

        Return:
            bytes: serialized ciphertext
        """
        cdef ostringstream outputter
        self._ptr_ctxt.save(outputter)
        return outputter.str()

    cpdef void from_file(self, fileName, encoding) except +:
        """from_file(str fileName, encoding)
        
        Alias of `load` with input sanitizer.

        Load the ciphertext from a file. Requires knowing the encoding.

        Args:
            fileName (str, pathlib.Path): path to file where the ciphertext is retrieved from.
            encoding (str, type, int, ENCODING_t): One of the following:

                * ('int', 'integer', int, 1, ENCODING_t.INTEGER) -> integer encoding.
                * ('float', 'double', float, 2, ENCODING_t.FRACTIONAL) -> fractional encoding.
                * ('array', 'batch', 'matrix', list, 3, ENCODING_t.BATCH) -> batch encoding.

        Return:
            None

        See Also:
            :func:`~Pyfhel.util.to_ENCODING_t`
        """
        self.load(_to_valid_file_str(fileName, check=True), encoding)

    cpdef void load(self, str fileName, encoding) except +:
        """load(self, str fileName, encoding)
        
        Load the ciphertext from a file.

        Args:
            fileName: (str) Valid file where the ciphertext is retrieved from.
            encoding (str, type, int, ENCODING_t): One of the following:

                * ('int', 'integer', int, 1, ENCODING_t.INTEGER) -> integer encoding.
                * ('float', 'double', float, 2, ENCODING_t.FRACTIONAL) -> fractional encoding.
                * ('array', 'batch', 'matrix', list, 3, ENCODING_t.BATCH) -> batch encoding.

              
        Return:
            None

        See Also:
            :func:`~Pyfhel.util.to_ENCODING_t`
        """
        cdef ifstream* inputter
        cdef string bFileName = fileName.encode('utf8')
        inputter = new ifstream(bFileName,binary)
        try:
            self._ptr_ctxt.load(deref(inputter))
        finally:
            del inputter
        self._encoding = to_ENCODING_t(encoding).value

    cpdef void from_bytes(self, bytes content, encoding) except +:
        """from_bytes(bytes content, encoding)

        Recover the serialized ciphertext from a binary/bytes string.

        Args:
            content (bytes):  Python bytes object containing the PyCtxt.
            encoding (str, type, int, ENCODING_t): One of the following:

                * ('int', 'integer', int, 1, ENCODING_t.INTEGER) -> integer encoding.
                * ('float', 'double', float, 2, ENCODING_t.FRACTIONAL) -> fractional encoding.
                * ('array', 'batch', 'matrix', list, 3, ENCODING_t.BATCH) -> batch encoding.

        Return:
            None

        See Also:
            :func:`~Pyfhel.util.to_ENCODING_t`
        """
        cdef stringstream inputter
        inputter.write(content,len(content))
        self._ptr_ctxt.load(inputter)
        self._encoding = to_ENCODING_t(encoding).value

            
    # =========================================================================
    # ============================= OPERATIONS ================================
    # =========================================================================          
    def __neg__(self):
        """__neg__()
        
        Negates this ciphertext.
        """
        self._pyfhel.negate(self)
        
    def __add__(self, other):
        """__add__(other)
        
        Sums this ciphertext with either another PyCtx or a PyPtxt plaintext.
        
        Sums with a PyPtxt/PyCtxt, storing the result a new ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Second summand.

        Returns:
            PyCtxt: Ciphertext resulting of substraction

        Raise:
            TypeError: if other doesn't have a valid type.

        See Also:
            :func:`~Pyfhel.Pyfhel.add`
        """
        if isinstance(other, PyCtxt):
            return self._pyfhel.add(self, other, in_new_ctxt=True)
        elif isinstance(other, PyPtxt):
            return self._pyfhel.add_plain(self, other, in_new_ctxt=True)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                return self._pyfhel.add_plain(self, other, in_new_ctxt=True)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                return self._pyfhel.add_plain(self, other, in_new_ctxt=True)
        else:
            raise TypeError("<Pyfhel ERROR> other summand must be either PyCtxt or PyPtxt")
    
    def __radd__(self, other): return self.__add__(other)
    def __iadd__(self, other):
        """Sums this ciphertext with either another PyCtx or a PyPtxt plaintext.
        
        Sums with a PyPtxt/PyCtxt, storing the result in this ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Second summand.
            
        Raise:
            TypeError: if other doesn't have a valid type.
        """
        if isinstance(other, PyCtxt):
            self._pyfhel.add(self, other, in_new_ctxt=False)
        elif isinstance(other, PyPtxt):
            self._pyfhel.add_plain(self, other, in_new_ctxt=False)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                self._pyfhel.add_plain(self, other, in_new_ctxt=False)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                self._pyfhel.add_plain(self, other, in_new_ctxt=False)
        else:
            raise TypeError("<Pyfhel ERROR> other summand must be either PyCtxt or PyPtxt")
        return self


    def __sub__(self, other):
        """__sub__(other)
        
        Substracts this ciphertext with either another PyCtxt or a PyPtxt plaintext.
        
        Substracts with a PyPtxt/PyCtxt, storing the result in a new ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Substrahend, to be substracted from this ciphertext.
        Returns:
            PyCtxt: Ciphertext resulting of substraction

        Raise:
            TypeError: if other doesn't have a valid type.
            
        See Also:
            :func:`~Pyfhel.Pyfhel.sub`
        """
        if isinstance(other, PyCtxt):
            return self._pyfhel.sub(self, other, in_new_ctxt=True)
        elif isinstance(other, PyPtxt):
            return self._pyfhel.sub_plain(self, other, in_new_ctxt=True)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                return self._pyfhel.sub_plain(self, other, in_new_ctxt=True)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                return self._pyfhel.sub_plain(self, other, in_new_ctxt=True)
        else:
            raise TypeError("<Pyfhel ERROR> substrahend must be either PyCtxt or PyPtxt")
    
    def __rsub__(self, other): return self.__sub__(other)
    def __isub__(self, other): 
        """Substracts this ciphertext with either another PyCtxt or a PyPtxt plaintext.
        
        Substracts with a PyPtxt/PyCtxt, storing the result in this ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Substrahend, to be substracted from this ciphertext.
            
        Raise:
            TypeError: if other doesn't have a valid type.
        """
        if isinstance(other, PyCtxt):
            self._pyfhel.sub(self, other, in_new_ctxt=False)
        elif isinstance(other, PyPtxt):
            self._pyfhel.sub_plain(self, other, in_new_ctxt=False)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                self._pyfhel.sub_plain(self, other, in_new_ctxt=False)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                self._pyfhel.sub_plain(self, other, in_new_ctxt=False)
        else:
            raise TypeError("<Pyfhel ERROR> substrahend must be either PyCtxt or PyPtxt")
        return self
                        
    def __mul__(self, other):
        """__mul__(other)
        
        Multiplies this ciphertext with either another PyCtxt or a PyPtxt plaintext.
        
        Multiplies with a PyPtxt/PyCtxt, storing the result in a new ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Multiplier, to be multiplied with this ciphertext.

        Returns:
            PyCtxt: Ciphertext resulting of multiplication

        Raise:
            TypeError: if other doesn't have a valid type.
            
        See Also:
            :func:`~Pyfhel.Pyfhel.multiply`
        """
        if isinstance(other, PyCtxt):
            return self._pyfhel.multiply(self, other, in_new_ctxt=True)
        elif isinstance(other, PyPtxt):
            return self._pyfhel.multiply_plain(self, other, in_new_ctxt=True)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                return self._pyfhel.multiply_plain(self, other, in_new_ctxt=True)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                return self._pyfhel.multiply_plain(self, other, in_new_ctxt=True)
        else:
            raise TypeError("<Pyfhel ERROR> multiplicand must be either PyCtxt, PyPtxt or int|float"
                            "(is %s instead)"%(type(other)))
     
    def __rmul__(self, other): return self.__mul__(other)
    def __imul__(self, other): 
        """Multiplies this ciphertext with either another PyCtxt or a PyPtxt plaintext.
        
        Multiplies with a PyPtxt/PyCtxt, storing the result in this ciphertext.

        Args:
            other (PyCtxt, PyPtxt): Multiplier, to be multiplied with this ciphertext.

        Returns:
            PyCtxt: Ciphertext resulting of multiplication

        Raise:
            TypeError: if other doesn't have a valid type.
        """
        if isinstance(other, PyCtxt):
            self._pyfhel.multiply(self, other, in_new_ctxt=False)
        elif isinstance(other, PyPtxt):
            self._pyfhel.multiply_plain(self, other, in_new_ctxt=False)
        elif isinstance(other, (int, float)):
            if self._encoding == ENCODING_t.INTEGER:
                other = self._pyfhel.encodeInt(int(other))
                self._pyfhel.multiply_plain(self, other, in_new_ctxt=False)
            elif self._encoding == ENCODING_t.FRACTIONAL:
                other = self._pyfhel.encodeFrac(float(other))
                self._pyfhel.multiply_plain(self, other, in_new_ctxt=False)
        else:
            raise TypeError("<Pyfhel ERROR> multiplicand must be either PyCtxt, PyPtxt or int|float"
                            "(is %s instead)"%(type(other)))
        return self


    def __truediv__(self, divisor):
        """__truediv__(divisor)
        
        Multiplies this ciphertext with the inverse of divisor.
        
        This operation can only be done with plaintexts. Division between 
        two Ciphertexts is not possible.

        For IntegerEncoding, the inverse is calculated as:
            inverse -> (divisor * inverse) mod p = 1

        For FractionalEncoding, the inverse is calculated as 1/divisor.

        Args:
            divisor (int, float, PyPtxt): divisor for the operation.
        """
        if isinstance(divisor, PyPtxt):
            divisor = divisor.decode()
        if not isinstance(divisor, (int, float)):
            raise TypeError("<Pyfhel ERROR> divisor must be float, int"
                            "or PyPtxt with those encodings (is %s:%s instead)"
                            %(str(divisor),type(divisor)))
        # Compute inverse. Int: https://stackoverflow.com/questions/4798654
        if self._encoding == ENCODING_t.INTEGER:
            divisor = int(divisor)
            p = self._pyfhel.getp()
            inverse = pow(divisor, p-2, p)
            inversePtxt = self._pyfhel.encodeInt(inverse)
        elif self._encoding == ENCODING_t.FRACTIONAL: # float. Standard inverse
            inverse = 1/float(divisor)
            inversePtxt = self._pyfhel.encodeFrac(inverse)
        else:
            raise TypeError("<Pyfhel ERROR> dividend encoding doesn't support"
                            "division (%s)"%(self._encoding))
        return self._pyfhel.multiply_plain(self, inversePtxt, in_new_ctxt=True)

    def __itruediv__(self, divisor):
        """Multiplies this ciphertext with the inverse of divisor.
        
        This operation can only be done with plaintexts. Division between 
        two Ciphertexts is not possible.

        For IntegerEncoding, the inverse is calculated as:
            inverse -> (divisor * inverse) mod p = 1

        For FractionalEncoding, the inverse is calculated as 1/divisor.

        Args:
            divisor (int, float, PyPtxt): divisor for the operation.
        """
        if isinstance(divisor, PyPtxt):
            divisor = divisor.decode()
        if not isinstance(divisor, (int, float)):
            raise TypeError("<Pyfhel ERROR> divisor must be float, int"
                            "or PyPtxt with those encodings (is %s:%s instead)"
                            %(str(divisor),type(divisor)))
        # Compute inverse. Int: https://stackoverflow.com/questions/4798654
        if self._encoding == ENCODING_t.INTEGER:
            divisor = int(divisor)
            p = self._pyfhel.getp()
            inverse = pow(divisor, p-2, p)
            inversePtxt = self._pyfhel.encodeInt(inverse)
        elif self._encoding == ENCODING_t.FRACTIONAL: # float. Standard inverse
            inverse = 1/float(divisor)
            inversePtxt = self._pyfhel.encodeFrac(inverse)
        else:
            raise TypeError("<Pyfhel ERROR> dividend encoding doesn't support"
                            "division (%s)"%(self._encoding))
        self._pyfhel.multiply_plain(self, inversePtxt, in_new_ctxt=False)
        return self

                                    
    def __pow__(self, exponent, modulo):
        """__pow__(exponent)
        
        Exponentiates this ciphertext to the desired exponent.
        
        Exponentiates to the desired exponent.

        Args:
            exponent (int): Exponent for the power.
            
        See Also:
            :func:`~Pyfhel.Pyfhel.power`
        """
        if(exponent==2):
            return self._pyfhel.square(self, in_new_ctxt=True)  
        else:
            return self._pyfhel.power(self, expon=exponent, in_new_ctxt=True)

    def __ipow__(self, exponent):
        """Exponentiates this ciphertext to the desired exponent, inplace.
        
        Exponentiates to the desired exponent.

        Args:
            exponent (int): Exponent for the power.
        """
        if(exponent==2):
            self._pyfhel.square(self, in_new_ctxt=False)  
        else:
            self._pyfhel.power(self, expon=exponent, in_new_ctxt=False)
        return self
                
    def __rshift__(self, k):
        """__rshift__(k)
        
        Rotates this ciphertext k positions to the right.
        Only works in batching mode.
        
        Args:
            k (int): Number of positions to rotate.
        """
        return self._pyfhel.rotate(self, -k, in_new_ctxt=True)

    def __irshift__(self, k):
        """Rotates this ciphertext k positions to the right, in-place.
        Only works in batching mode.

        Args:
            k (int): Number of positions to rotate.
        """
        self._pyfhel.rotate(self, -k, in_new_ctxt=False)
        return self

    def __lshift__(self, k):
        """Rotates this ciphertext k positions to the left.
        Only works in batching mode.

        Args:
            k (int): Number of positions to rotate.
        """
        return self._pyfhel.rotate(self, k, in_new_ctxt=True)

    def __ilshift__(self, k):
        """Rotates this ciphertext k positions to the left, in-place.
        Only works in batching mode.

        Args:
            k (int): Number of positions to rotate.
        """
        self._pyfhel.rotate(self, k, in_new_ctxt=False)
        return self

    def __invert__(self):
        """__invert__()
        
        Relinarizes this ciphertext in-place.

        Requires valid relinearization keys with a bitcount higher than the
        current size of this ciphertext.

        See Also:
            :func:`~Pyfhel.Pyfhel.relinearize`
        """
        self._pyfhel.relinearize(self)
        return self

    # =========================================================================
    # ============================ ENCR/DECR/CMP ==============================
    # =========================================================================

    def __len__(self):
        """__len__()
        
        Return the current size of the ciphertext.
        
        See Also:
            :func:`~Pyfhel.PyCtxt.size`
        """
        return self.size()

    # def __int__(self):
    #     if (self._encoding != ENCODING_T.INTEGER):
    #         raise RuntimeError("<Pyfhel ERROR> casting (decrypting) to int requires INTEGER encoding")
    #     return self._pyfhel.decryptInt(self)

    # def __float__(self):
    #     if (self._encoding != ENCODING_T.FRACTIONAL):
    #         raise RuntimeError("<Pyfhel ERROR> casting (decrypting) to float requires FRACTIONAL encoding")
    #     return self._pyfhel.decryptFrac(self)
    
    def __repr__(self):
        """__repr__()
        
        Prints information about the current ciphertext"""
        sk_not_empty = self._pyfhel is not None and not self._pyfhel.is_secretKey_empty()
        return "<Pyfhel Ciphertext at {}, encoding={}, size={}/{}, noiseBudget={}>".format(
                hex(id(self)),
                ENCODING_t(self._encoding).name,
                self.size(),
                self.size_capacity(),
                self._pyfhel.noiseLevel(self) if sk_not_empty else "-"
                  )
                
    def __bytes__(self):
        """__bytes__()
        
        Serialize current ciphertext to bytes"""
        return self.to_bytes()

    def encrypt(self, value):
        """encrypt(value)
        
        Encrypts the given value using _pyfhel.
        
        Arguments:
            value (int, float, np.array): Encrypts accordingly to the tipe
            
        Return:
            None
            
        See Also:
            :func:`~Pyfhel.Pyfhel.encrypt`
        """
        self._pyfhel.encrypt(value, self)
    
    def decrypt(self):
        """decrypt()
        
        Decrypts itself using _pyfhel.
        
        Arguments:
            None
            
        Return:
            int, float, np.array: value decrypted.
   
        See Also:
            :func:`~Pyfhel.Pyfhel.decrypt`
        """
        return self._pyfhel.decrypt(self, decode_value=True)
