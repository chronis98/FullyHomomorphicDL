/*
 * @file matrixser.cpp - matrix serializations operations.
 * @author  TPOC: palisade@njit.edu
 *
 * @copyright Copyright (c) 2017, New Jersey Institute of Technology (NJIT)
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef _SRC_LIB_CORE_MATH_MATRIXSER_CPP
#define _SRC_LIB_CORE_MATH_MATRIXSER_CPP

#include "../utils/serializablehelper.h"
#include "../lattice/field2n.h"
#include "matrix.cpp"
#include "matrixstrassen.h"
using std::invalid_argument;

// this is the serializations of matrices
// not implemented, so just placeholders

namespace lbcrypto {

MATRIX_NOT_SERIALIZABLE(double)
MATRIX_NOT_SERIALIZABLE(int)
MATRIX_NOT_SERIALIZABLE(int64_t)
MATRIX_NOT_SERIALIZABLE(uint64_t)
MATRIX_NOT_SERIALIZABLE(Field2n)


// TODO implement!
#ifdef OUT
template<>
bool Matrix<Poly>::Serialize(Serialized* serObj) const {

  if( !serObj->IsObject() ){
    serObj->SetObject();
  }
  
  //SerializeVectorOfVector("Matrix", elementName<Element>(), this->data, serObj);
  
  //std::cout << typeid(Element).name() << std::endl;
  bool rc = false;
    for( size_t r=0; r<rows; r++ ) {
      for( size_t c=0; c<cols; c++ ) {
	rc = data[r][c].Serialize(serObj);
	if (!rc) {
	  PALISADE_THROW(lbcrypto::serialize_error ,"serialization failure in Matrix<Poly> "
			 +to_string(r)+", "+to_string(c));
	}
      }
    }
    return true;
}

template<>
bool Matrix<Poly>::Deserialize(const Serialized& serObj) {
    std::cout<<"Matrix<Poly>::Deserialize() not written"<<std::endl;
    return false;
}

template<>
bool Matrix<NativePoly>::Serialize(Serialized* serObj) const {
 if( !serObj->IsObject() ){
    serObj->SetObject();
  }

  bool rc = false;
  for( size_t r=0; r<rows; r++ ) {
    for( size_t c=0; c<cols; c++ ) {
      data[r][c].Serialize(serObj);
      if (!rc) {
	PALISADE_THROW(lbcrypto::serialize_error ,"serialization failure in Matrix<NativePoly> "
		       +to_string(r)+", "+to_string(c));
      }
    }
  }
  return true;
}

template<>
bool Matrix<NativePoly>::Deserialize(const Serialized& serObj) {
      std::cout<<"Matrix<NativePoly>::Deserialize() not written"<<std::endl;
	return false;
}

template<>
bool Matrix<DCRTPoly>::Serialize(Serialized* serObj) const {
  std::cout <<" Matrix<DCRTPoly>::Serialize, unsupported, use SerializeMatrix"<<std::endl;
  if( !serObj->IsObject() ){
    serObj->SetObject();
  }
  bool rc = false;
  for( size_t r=0; r<rows; r++ ) {
    for( size_t c=0; c<cols; c++ ) {
      rc = data[r][c].Serialize(serObj); //call the serialization for this matrix
      if( !rc ) {
	PALISADE_THROW(lbcrypto::serialize_error,
		       "Matrix<DCRTPoly>::Serialize error element "
		       +to_string(r)+", "+to_string(c));
	return false;
      }
    }
  }
  return true;
}

template<>
bool Matrix<DCRTPoly>::Deserialize(const Serialized& serObj) {
    std::cout <<" Matrix<DCRTPoly>::DeSerialize, not written"<<std::endl;

  return false;
}

template<>
bool MatrixStrassen<Poly>::Serialize(Serialized* serObj) const {
  std::cout<<"MatrixStrassen<Poly>::Serialize() not written"<<std::endl;
  return false;
}

template<>
bool MatrixStrassen<Poly>::Deserialize(const Serialized& serObj) {
  std::cout<<"MatrixStrassen<Poly>::Deserialize() not written"<<std::endl;
  return false;
}

#endif

}

#endif
