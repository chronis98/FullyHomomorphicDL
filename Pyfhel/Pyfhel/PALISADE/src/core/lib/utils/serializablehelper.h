
/**
 * @file serializablehelper.h Helper methods for serialization.
 * @author  TPOC: palisade@njit.edu
 *
 * @copyright Copyright (c) 2017, New Jersey Institute of Technology
 * (NJIT) All rights reserved.  Redistribution and use in source and
 * binary forms, with or without modification, are permitted provided
 * that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */ 

#ifndef LBCRYPTO_SERIALIZABLEHELPER_H
#define LBCRYPTO_SERIALIZABLEHELPER_H

#include "serializable.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/error/en.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <map>
#include <iterator>
#include <algorithm>

#include "../math/backend.h"
#include "../lattice/backend.h"
#include "../math/matrix.h"

#define RAPIDJSON_NO_SIZETYPEDEFINE

namespace lbcrypto {

  class SerializableHelper {

  public:
    /**
     * Generates a std::string for a serialized Palisade object (a
     * rapidjson Document)
     * @param serObj the serialized Palisade object
     * @param jsonString - string with the JSON data structure of the
     * serialized Palisade object.
     * @return success or failure
     */
    static bool SerializationToString(const Serialized& serObj, std::string& jsonString);

    /**
     * Generates a Pretty std::string for a serialized Palisade object (a 
     * rapidjson Document)
     * @param serObj the serialized Palisade object
     * @param jsonString - string with the JSON data structure of the 
     * serialized Palisade object.
     * @return success or failure
     */
    static bool SerializationToPrettyString(const Serialized& serObj, std::string& jsonString);

    /**
     * Writes serialization to a file
     * @param serObj - the serialized Palisade object
     * @param out - destination stream for writing the serialization
     * @return success or failure
     */
    static bool SerializationToStream(const Serialized& serObj, std::ostream& out);

    /**
     * Generates a serialized Palisade object from an input JSON string
     * @param jsonString reflecting the nested JSON data structure of the 
     * serialized Palisade object.
     * @param serObj stores the serialized Palisade object's attributes.
     * @return success or failure
     */
    static bool StringToSerialization(const std::string& jsonString, Serialized* serObj);

    /**
     * Generate a serialized Palisade object from a JSON string read 
     * from an input stream
     * @param in - stream to read
     * @param serObj stores the serialized Palisade object's attributes.
     * @return success or failure
     */
    static bool StreamToSerialization(std::istream& in, Serialized* serObj);

    /**
     * Saves a serialized Palisade object's JSON string to file
     * @param serObj is the serialized object
     * @param outputFileName is the name of the file to save JSON data string to.
     * @return success or failure
     */
    static bool WriteSerializationToFile(const Serialized& serObj, std::string outputFileName);

    /**
     * Saves a pretty serialized Palisade object's JSON string to file
     * @param serObj is the serialized object
     * @param outputFileName is the name of the file to save JSON data string to.
     * @return success or failure
     */
    static bool WriteSerializationToPrettyFile(const Serialized& serObj, std::string outputFileName);

    /**
     * Read a serialized Palisade object from a JSON file
     * @param jsonFileName is the file to read in for the Palisade object's 
     * nested serialized JSON data structure.
     * @param map containing the serialized object read from the file
     * @return success or failure
     */
    static bool ReadSerializationFromFile(const std::string jsonFileName, Serialized* map, bool verbose=false);

  };

  /** 
   * Helper template Adds the contents of an STL vector<foo> to 
   *  a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize() method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param &inVector the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */

  template<typename T>
    void SerializeVector(const std::string& vectorName, const std::string& typeName,
			 const std::vector<T> &inVector, Serialized* serObj) {
    //make sure the input is a rapidjson object
    if( ! serObj->IsObject() )
      serObj->SetObject();

    Serialized ser(rapidjson::kObjectType, &serObj->GetAllocator());
    ser.AddMember("Container", "Vector", serObj->GetAllocator());
    ser.AddMember("Typename", typeName, serObj->GetAllocator());
    ser.AddMember("Length", std::to_string(inVector.size()), serObj->GetAllocator());

    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());
    for( size_t i=0; i<inVector.size(); i++ ) {
      bool rc;
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      rc = inVector[i].Serialize(&oneEl);
      if (!rc) {
	PALISADE_THROW(lbcrypto::serialize_error,
		       "SerializeVector<"+typeName+"> element "
		       +to_string(i)+" serialilzation failed.");
      }
      SerialItem key( std::to_string(i), serObj->GetAllocator() );
      serElements.AddMember(key, oneEl, serObj->GetAllocator());
    }

    ser.AddMember("Members", serElements, serObj->GetAllocator());

    serObj->AddMember(SerialItem(vectorName, serObj->GetAllocator()), ser, serObj->GetAllocator());
  }


  /** 
   * Helper template Adds the contents of an STL vector<shared_ptr<foo>> to 
   *  a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize() method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param &inVector the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */
 
  template<typename T>
    void SerializeVectorOfPointers(const std::string& vectorName, const std::string& typeName, const std::vector<shared_ptr<T>> &inVector, Serialized* serObj) {

    //make sure the input is a rapidjson object
    if( ! serObj->IsObject() )
      serObj->SetObject();
    
    Serialized ser(rapidjson::kObjectType, &serObj->GetAllocator());
    ser.AddMember("Container", "VectorOfPointer", serObj->GetAllocator());
    ser.AddMember("Typename", typeName, serObj->GetAllocator());
    ser.AddMember("Length", std::to_string(inVector.size()), serObj->GetAllocator());

    bool rc = false;
    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());
    for( size_t i=0; i<inVector.size(); i++ ) {
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      rc = inVector[i]->SerializeWithoutContext(&oneEl);
      if (!rc) {
	PALISADE_THROW(lbcrypto::serialize_error,
		       "SerializeVectorOfPointers<"+typeName+"> element "
		       +to_string(i)+" SerialilzeWithoutContext failed.");
      }
      SerialItem key( std::to_string(i), serObj->GetAllocator() );
      serElements.AddMember(key, oneEl.Move(), serObj->GetAllocator());
    }

    ser.AddMember("Members", serElements.Move(), serObj->GetAllocator());

    serObj->AddMember(SerialItem(vectorName, serObj->GetAllocator()), ser, serObj->GetAllocator());
  }

  /** 
   * Helper template Adds the contents of an STL map<bar, shared_ptr<foo>> to 
   *  a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize() method to serialize.
   * @param mapName of map to be serialized
   * @param typeName of element within the map
   * @param &inMap the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */
  template<typename K, typename T>
    void SerializeMapOfPointers(const std::string& mapName, const std::string& typeName, const std::map<K,shared_ptr<T>> inMap, Serialized* serObj) {

    //make sure the input is a rapidjson object
    if( ! serObj->IsObject() )
      serObj->SetObject();

    Serialized ser(rapidjson::kObjectType, &serObj->GetAllocator());
    ser.AddMember("Container", "MapOfPointer", serObj->GetAllocator());
    ser.AddMember("Typename", typeName, serObj->GetAllocator());

    ser.AddMember("Size", std::to_string(inMap.size()), serObj->GetAllocator());

      
    bool rc = false;
    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());

    for( const auto& kv : inMap ) {
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      rc = kv.second->SerializeWithoutContext(&oneEl);


      string kvstring = to_string(kv.first);
      if (!rc) {
	PALISADE_THROW(lbcrypto::serialize_error,
		       "SerializeMapOfPointers<"+typeName+"> map element "
		       //+to_string(kv.first)+" SerialilzeWithoutContext failed.");
		       +kvstring+" SerialilzeWithoutContext failed.");
      }

      //SerialItem key(to_string(kv.first), serObj->GetAllocator() );
      SerialItem key(kvstring, serObj->GetAllocator() );
      serElements.AddMember(key, oneEl.Move(), serObj->GetAllocator());
    }

    ser.AddMember("Members", serElements.Move(), serObj->GetAllocator());

    serObj->AddMember(SerialItem(mapName, serObj->GetAllocator()), ser, serObj->GetAllocator());
  }


  /** 
   * Helper template Adds the contents of an STL vector<vector<shared_ptr<Matrix<foo>>>>
   * to a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector of vectors
   * @param inVector the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */
 
  template<typename T>
    void SerializeVectorOfVectorOfPointersToMatrix(const std::string& vectorName, const std::string& typeName, const std::vector<vector<shared_ptr<Matrix<T>>>> &inVector, Serialized* serObj) {
    // strategy: serialize a vector containing VpM,
    // use VpM  serialization helper for each element
    
    //make sure the input is a rapidjson object
    bool dbg_flag = false;
    TimeVar t1; //for TIC TOC
    
    if( ! serObj->IsObject() )
      serObj->SetObject();
    
    //make top level member
    Serialized topser(rapidjson::kObjectType, &serObj->GetAllocator());
    //add top member components
    topser.AddMember("Container", "VectorOfVectorOfPointerToMatrix", serObj->GetAllocator());
    topser.AddMember("Typename", typeName, serObj->GetAllocator());
    topser.AddMember("Length", std::to_string(inVector.size()), serObj->GetAllocator());

    // make member container for all elements
    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());

    for( size_t i=0; i<inVector.size(); i++ ) {//for each element
      TIC(t1);
      //serialize the ith element
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      
      //inVector[i]->Serialize(&oneEl);
      std::string elName  = vectorName+"_"+std::to_string(i);
      SerializeVectorOfPointersToMatrix(elName, typeName, inVector[i], &oneEl);

      //add it with the index as a key to the member container
      SerialItem key( std::to_string(i), serObj->GetAllocator() );
      serElements.AddMember(key, oneEl.Move(), serObj->GetAllocator());
      DEBUG("vvpm "<<i<<" serial time "<<TOC(t1));
    }

    //add the member container to the top level
    topser.AddMember("Members", serElements.Move(), serObj->GetAllocator());

    //add the top level to the inpupt serial item
    serObj->AddMember(SerialItem(vectorName, serObj->GetAllocator()), topser, serObj->GetAllocator());
  }



  /** 
   * Helper template Adds the contents of an STL vector<shared_ptr<Matrix<foo>>>
   * to a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param inVector the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */
  template<typename T>
    void SerializeVectorOfPointersToMatrix(const std::string& vectorName, const std::string& typeName, const std::vector<shared_ptr<Matrix<T>>> &inVector, Serialized* serObj) {
    bool dbg_flag = false;
    TimeVar t1; //for TIC TOC
    

    //make sure the input is a rapidjson object
    if( ! serObj->IsObject() )
      serObj->SetObject();
    
    //make top level member
    Serialized topser(rapidjson::kObjectType, &serObj->GetAllocator());
    //add top member components
    topser.AddMember("Container", "VectorOfPointerToMatrix", serObj->GetAllocator());
    topser.AddMember("Typename", typeName, serObj->GetAllocator());
    topser.AddMember("Length", std::to_string(inVector.size()), serObj->GetAllocator());

    // make member container for all elements
    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());

    for( size_t i=0; i<inVector.size(); i++ ) {//for each element
      TIC(t1);
  
      //serialize the ith element
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      
      //inVector[i]->Serialize(&oneEl);
      std::string elName  = "Matrix_"+std::to_string(i);
      SerializeMatrix(elName, typeName, *inVector[i], &oneEl);

      //add it with the index as a key to the member container
      SerialItem key( std::to_string(i), serObj->GetAllocator() );
      serElements.AddMember(key, oneEl.Move(), serObj->GetAllocator());
      DEBUG("        vpm "<<i<<" serial time "<<TOC(t1));
    }

    //add the member container to the top level
    topser.AddMember("Members", serElements.Move(), serObj->GetAllocator());

    //add the top level to the inpupt serial item
    serObj->AddMember(SerialItem(vectorName, serObj->GetAllocator()), topser, serObj->GetAllocator());
  }

  /** 
   * Helper template Adds the contents of an STL vector<Matrix<foo>>
   * to a serialized Palisade object as a nested JSON data structure
   * foo must be a pointer to a serializable object as the function uses the 
   * foo->Serialize method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param inVector the STL vector to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */
  
  template<typename T>
    void SerializeVectorOfMatrix(const std::string& vectorName, const std::string& typeName, const std::vector<Matrix<T>> &inVector, Serialized* serObj) {

    //make sure the input is a rapidjson object
    if( ! serObj->IsObject() )
      serObj->SetObject();
    
    //make top level member
    Serialized topser(rapidjson::kObjectType, &serObj->GetAllocator());
    //add top member components
    topser.AddMember("Container", "VectorOfMatrix", serObj->GetAllocator());
    topser.AddMember("Typename", typeName, serObj->GetAllocator());
    topser.AddMember("Length", std::to_string(inVector.size()), serObj->GetAllocator());

    // make member container for all elements
    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());

    for( size_t i=0; i<inVector.size(); i++ ) {//for each element
      //serialize the ith element
      Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
      
      //inVector[i]->Serialize(&oneEl);
      std::string elName  = "Matrix_"+std::to_string(i);
      SerializeMatrix(elName, typeName, inVector[i], &oneEl);

      //add it with the index as a key to the member container
      SerialItem key( std::to_string(i), serObj->GetAllocator() );
      serElements.AddMember(key, oneEl.Move(), serObj->GetAllocator());
    }

    //add the member container to the top level
    topser.AddMember("Members", serElements.Move(), serObj->GetAllocator());

    //add the top level to the inpupt serial item
    serObj->AddMember(SerialItem(vectorName, serObj->GetAllocator()), topser, serObj->GetAllocator());
  }

  /** 
   * Helper template fills  an STL vector<foo> with the contents of a 
   *  a serialized Palisade object made with SerializeVector
   * foo must be a serializable object as the function uses the 
   * foo.Deserialize() method to deserialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param *outVector pointer to the STD vector to be deserialized
   * @param it an iterator into the serial object to be deserialisesd
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return
  
  template<typename T>
    bool DeserializeVector(const std::string& vectorName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, std::vector<T>* outVector) {
    bool dbg_flag = false;
    std::string fname = "DeserializeVector<"+typeName+"> ";
    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }

    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "Wrong type name found: "+ string(mIt->value.GetString())
	    + "expected :" +typeName );
    }
    mIt = it->value.FindMember("Length");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Length");
    }

    outVector->clear();
    outVector->resize( std::stoi(mIt->value.GetString()) );

    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ){
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    }
    const SerialItem& members = mIt->value;

    for( size_t i=0; i<outVector->size(); i++ ) {
      Serialized::ConstMemberIterator eIt = members.FindMember( std::to_string(i) );
      if( eIt == members.MemberEnd() ) {
	PALISADE_THROW(lbcrypto::deserialize_error, "could not find index "+to_string(i));
      }

      T vectorElem;
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(typeName);
      if( s2 == eIt->value.MemberEnd() ){
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find typename "+ typeName+ "for index "+to_string(i));
      }
      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      DEBUGEXP(i);
      if (s2->value.IsString()) {
	DEBUGEXP(s2->value.GetString());
      }
      if (s2->value.IsUint64()){ 
	DEBUGEXP(s2->value.GetUint64());
      }
      ser.AddMember(k, v, ser.GetAllocator());
      if( vectorElem.Deserialize(ser) ) {
	DEBUG("Deserialized "<< vectorElem);
	outVector->at(i) = vectorElem;
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error, "Deserialization failed ");
      }	
    }
    return true;
  }
  
  
  /** 
   * Helper template Fills an STL vector<shared_ptr<foo>> with the contents of a 
   *  a serialized Palisade object made with SerializeVectorOfPointers
   * foo must be a pointer to a serializable object as the function uses the 
   * foo->SerializeWithoutContext() method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param inMap the STL map to be deserialized
   * @param it an iterator into the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return

  template<typename T>
    bool DeserializeVectorOfPointers(const std::string& vectorName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, std::vector<shared_ptr<T>>* outVector) {

    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }

    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" + typeName );
    }
    mIt = it->value.FindMember("Length");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Length");
    };

    outVector->clear();
    outVector->resize( std::stoi(mIt->value.GetString()) );

    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    };

    const SerialItem& members = mIt->value;

    for( size_t i=0; i<outVector->size(); i++ ) {
      Serialized::ConstMemberIterator eIt = members.FindMember( std::to_string(i) );
      if( eIt == members.MemberEnd() ) {
	  PALISADE_THROW(lbcrypto::deserialize_error,
			 "could not find Vector entry "+to_string(i));
      };

      T vectorElem;
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(typeName);

      if( s2 == eIt->value.MemberEnd() ) {
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find "+string(typeName)+" in vector");
      };

      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      ser.AddMember(k, v, ser.GetAllocator());

      if( vectorElem.Deserialize(ser) ) {
	outVector->at(i).reset( new T(vectorElem) );
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not deserialize Vector entry "+to_string(i));
      }
    }

    return true;
  }


    /** 
   * Helper template Fills an STL map<K,shared_ptr<foo>> with the contents of a 
   *  a serialized Palisade object made with SerializeMapOfPointers
   * foo must be a pointer to a serializable object as the function uses the 
   * foo->SerializeWithoutContext() method to serialize.
   * @param mapName name of map 
   * @param typeName of element within the map
   * @param inMap the STL map to be deserialized
   * @param it an iterator into the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return

  template<typename K, typename T>
    bool DeserializeMapOfPointers(const std::string& mapName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, std::map<K, shared_ptr<T>>* outMap) {
    bool dbg_flag = false;

    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }

    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" + typeName );
    }
    mIt = it->value.FindMember("Size");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Size");
    };
    usint checkSize = std::stoi(mIt->value.GetString()); 


    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    };

    const SerialItem& members = mIt->value;

    // need to iterate over the map 
    for (Serialized::ConstMemberIterator eIt = members.MemberBegin(); eIt !=members.MemberEnd(); ++eIt){
      auto key = eIt->name.GetString();
      DEBUGEXP(eIt->name.GetString());
      DEBUGEXP(string(key));

      auto pMapElem = make_shared<T>();
      
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(typeName);

      if( s2 == eIt->value.MemberEnd() ) {
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find "+string(typeName)+" in map");
      };

      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      ser.AddMember(k, v, ser.GetAllocator());


      if( pMapElem->Deserialize(ser) ) {
	(*outMap)[atoi(key)] = pMapElem;
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not deserialize Map entry "+string(key));
      }
    }

    if (checkSize != outMap->size() ) {
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not deserialize "+to_string(checkSize)+" Map entries, found only  "+to_string(outMap->size()));
    }      
    
    return true;
  }

  /** 
   * Helper template Adds the contents of a palisade Matrix<foo> to 
   *  a serialized Palisade object as a nested JSON data structure
   * foo must be a serializable object as the function uses the 
   * foo.Serialize() method to serialize.
   * @param matrixName 
   * @param typeName of element within the matrix
   * @param inMatrix the Palisade matrix to be serialized
   * @param *serObj the serial object to be modfied, if not a serial object
   * then it is made a serial object
   * throws a Palisade serialize_error on error
   * @return void  
   */

 
  template<typename T>
    void SerializeMatrix(const std::string &matrixName, const std::string &typeName, const Matrix<T> &inMatrix, Serialized* serObj) {
    
    bool dbg_flag = false;
    DEBUG("in SerializeMatrix<"<<typeName<<">");
    //make sure the input is a rapidjson object
    TimeVar t1; //for TIC TOC
    
 
    if( ! serObj->IsObject() )
      serObj->SetObject();

        size_t rows = inMatrix.GetRows();
    size_t cols = inMatrix.GetCols();

    Serialized ser(rapidjson::kObjectType, &serObj->GetAllocator());
    ser.AddMember("Container", "Matrix", serObj->GetAllocator());
    ser.AddMember("Typename", typeName, serObj->GetAllocator());


    ser.AddMember("NumRows", std::to_string(rows), serObj->GetAllocator());
    ser.AddMember("NumColumns", std::to_string(cols), serObj->GetAllocator());

    Serialized serElements(rapidjson::kObjectType, &serObj->GetAllocator());

    for( size_t i=0; i<rows; i++ ) {
      for( size_t j=0; j<cols; j++ ) {
	bool rc = 0;
	TIC(t1);
	Serialized oneEl(rapidjson::kObjectType, &serObj->GetAllocator());
	rc = ((inMatrix.GetData())[i][j]).Serialize(&oneEl);
	if (!rc) {
	  PALISADE_THROW(lbcrypto::serialize_error,
			 "SerializeMatrix<"+typeName+"> element "
			 +to_string(i)+", "+to_string(j)+" serialilzation failed.");
	}
	std::string keystring =std::to_string(i)
	  + "," + std::to_string(j);
			
	SerialItem key(keystring , serObj->GetAllocator() );
	serElements.AddMember(key, oneEl, serObj->GetAllocator());
	DEBUG("              m "<<i<<","<<j<<" serial time "<<TOC(t1));
	
      }
    }

    ser.AddMember("Members", serElements, serObj->GetAllocator());

    serObj->AddMember(SerialItem(matrixName, serObj->GetAllocator()), ser, serObj->GetAllocator());
  }
  
  /** 
   * Helper template fills  an Palisade Matrix<foo> with the contents of a 
   *  a serialized Palisade object made with SerializeMatrix<foo>
   * foo must be a serializable object as the function uses the 
   * foo.Deserialize() method to deserialize.
   * @param MatrixName input matrix name 
   * @param typeName of element within the matrix<
   * @param inMatrix the palisade matrix to be deserialized
   * @param *serObj the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return
  
  template<typename T>
    bool DeserializeMatrix(const std::string& MatrixName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, Matrix<T>* outMatrix) {
    bool dbg_flag = false;
    DEBUG("in DeserializeMatrix");
    DEBUGEXP(MatrixName);
    DEBUGEXP(typeName);
    
    
    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }
    
    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" +typeName );
    }

    mIt = it->value.FindMember("NumRows");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find NumRows");
    }
    size_t nrows = std::stoi(mIt->value.GetString());
    
    mIt = it->value.FindMember("NumColumns");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find NumColumns");
    }
    
    size_t ncols = std::stoi(mIt->value.GetString());
    
    outMatrix->SetSize(0,0);
    outMatrix->SetSize(nrows, ncols);
    
    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ){
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    }
    const SerialItem& members = mIt->value;
    
    //loop over entire matrix
    for( size_t i=0; i<nrows; i++ ) {
      for( size_t j=0; j<ncols; j++ ) {
	
	std::string keystring =std::to_string(i)
	  + "," + std::to_string(j);
	
	Serialized::ConstMemberIterator eIt = members.FindMember(keystring);
	if( eIt == members.MemberEnd() ) {
	  PALISADE_THROW(lbcrypto::deserialize_error,
			 "could not find Matrix entry "+to_string(i)+ ", "+to_string(j));
	}
	
	T matrixElem;
	SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(typeName);
	if( s2 == eIt->value.MemberEnd() ){
	  PALISADE_THROW(lbcrypto::deserialize_error,
			 "could not find typename "+ typeName+ "for "
			 +to_string(i)+ ", "+to_string(j));
	}
	Serialized ser(rapidjson::kObjectType);
	SerialItem k( typeName, ser.GetAllocator() );
	SerialItem v( s2->value, ser.GetAllocator() );

	ser.AddMember(k, v, ser.GetAllocator());
	if( matrixElem.Deserialize(ser) ) {
	  //DEBUG("Deserialized "<< matrixElem);
	  (*outMatrix)(i,j) = matrixElem;
	} else {
	  PALISADE_THROW(lbcrypto::deserialize_error,
			 "Deserialization of "+to_string(i)+", "+to_string(j)+" failed ");
	}	
      }
    }

    DEBUGEXP((*outMatrix).GetRows());
    DEBUGEXP((*outMatrix).GetCols());
    DEBUGEXP((*outMatrix)(0,0));

    return true;
  }
  
  /** 
   * Helper template Fills an STL vector<shared_ptr<matrix<foo>>> with the contents of a 
   *  a serialized Palisade object made with SerializeVectorOfPointers
   * foo must be a serializable object as the function uses the 
   * foo.DeSerialize method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param outVector the STL vector to contain the result
   * @param it an iterator into the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return
  
  template<typename T>
    bool DeserializeVectorOfPointersToMatrix(const std::string& vectorName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, std::vector<shared_ptr<Matrix<T>>>* outVector) {
    bool dbg_flag = false;

    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }

    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" + typeName );
    }
    mIt = it->value.FindMember("Length");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Length");
    };

    outVector->clear();
    outVector->resize( std::stoi(mIt->value.GetString()) );

    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    };

    const SerialItem& members = mIt->value;
    
    //loop over the entire vector 
    for( size_t i=0; i<outVector->size(); i++ ) {
      std::string keystring =std::to_string(i);
 
      Serialized::ConstMemberIterator eIt = members.FindMember(keystring);
      if( eIt == members.MemberEnd() ) {
	  PALISADE_THROW(lbcrypto::deserialize_error,
			 "could not find Vector entry "+to_string(i));
      };

      //make a shart pointer to an empty Matrix<T>
      auto pT = make_shared<Matrix<T>>([](){ return T(); }, 0,0);

      //within the key's member, find the sub member with the typename
      //and point to it with s2.
      string matrix_name = "Matrix_"+to_string(i);
      DEBUG(" Searching for "<<matrix_name);
      
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(matrix_name);
      if( s2 == eIt->value.MemberEnd() ){
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find matrix name "+ matrix_name);
      }
      DEBUG("Found "<<matrix_name);
      
      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      DEBUGEXP(i);
      if (s2->value.IsString()) {
	DEBUGEXP(s2->value.GetString());
      }
      if (s2->value.IsUint64()){ 
	DEBUGEXP(s2->value.GetUint64());
      }
      ser.AddMember(k, v, ser.GetAllocator());
      
      //deserialize the matrix in s2
      std::string matname = "Matrix";
      std::string elem_name = typeName;
      bool rc = DeserializeMatrix(matname, elem_name, s2, pT.get());
      if(rc) {
	DEBUG("Deserialized matrix at index "<<i);
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error, "Deserialization of Matrix "+to_string(i)+" failed internally");
      }
      outVector->at(i)=pT; //store the pointer to the Matrix<T> into the vector location
    }

    return true;
  }


  /** 
   * Helper template Fills an STL vector<vector<shared_ptr<matrix<foo>>>> with the contents of a 
   *  a serialized Palisade object made with SerializeVectorOfVectorOfPointersToMatrix()
   * foo must be a serializable object as the function uses the 
   * foo.DeSerialize method to serialize.
   * @param vectorName 
   * @param typeName of element within the vector
   * @param outVector the STL  Vector to contain the result 
   * @param it an iterator into the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return

  template<typename T>
    bool DeserializeVectorOfVectorOfPointersToMatrix(const std::string& MatrixName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, vector<vector<shared_ptr<Matrix<T>>>>*outVector) {
   
    bool dbg_flag = false;

    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }
   
    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" +typeName );
    }
   
    mIt = it->value.FindMember("Length");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Length");
    }
    size_t length = std::stoi(mIt->value.GetString());
    
    outVector->clear();
    outVector->resize(length);
   
    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ){
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    }
    const SerialItem& members = mIt->value;
    DEBUG("looping over vector");
    //loop over entire vector
    for( size_t i=0; i<outVector->size(); i++ ) {
      std::string keystring =std::to_string(i);
     
      Serialized::ConstMemberIterator eIt = members.FindMember(keystring);
      if( eIt == members.MemberEnd() ) {
	PALISADE_THROW(lbcrypto::deserialize_error, "could not find Vector entry "+to_string(i));
      }

      string el_name = MatrixName+"_"+to_string(i);
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(el_name);
      if( s2 == eIt->value.MemberEnd() ){
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find element name "+ el_name);
      }
     
      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      DEBUGEXP(i);
      if (s2->value.IsString()) {
	DEBUGEXP(s2->value.GetString());
      }
      if (s2->value.IsUint64()){ 
	DEBUGEXP(s2->value.GetUint64());
      }
      ser.AddMember(k, v, ser.GetAllocator());
      std::string matname = "Matrix";
      std::string elem_name = typeName;
      bool rc = DeserializeVectorOfPointersToMatrix(matname, elem_name, s2, &(outVector->at(i)));
      if(rc) {
	DEBUG("Deserialized VectorOfPointersToMatrix at element"<<i );
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error, "Deserialization of VectorOfPointersToMatrix "+to_string(i)+" failed internally");
      }	
    }
    return true;
  }

  /** 
   * Helper template Fills an STL vector<<matrix<foo>> with the contents of a 
   *  a serialized Palisade object made with SerializeVectorOfMatrix()
   * foo must be a serializable object as the function uses the 
   * foo.DeSerialize method to serialize.
   * @param vectorName name of vector
   * @param typeName of element within the matrix
   * @param outVector the STL  Vector to contain the result 
   * @param it an iterator into the serial object to be deserialised
   * throws a Palisade deserialize_error on error
   * @return true if successful 
   */

  //todo: should be made a void return

  template<typename T>
    bool DeserializeVectorOfMatrix(const std::string& VectorName, const std::string& typeName, const SerialItem::ConstMemberIterator& it, vector<Matrix<T>>* outVector /*, std::function<unique_ptr<T>(void)> alloc_function */) {
   
    bool dbg_flag = false;
    
    DEBUG("Searching for Typename");
    SerialItem::ConstMemberIterator mIt = it->value.FindMember("Typename");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Typename  ");
    }

    DEBUG("Searching for "<<typeName);   
    if( mIt->value.GetString() != typeName ) {
      PALISADE_THROW(lbcrypto::deserialize_error,
		     "Wrong type name found: "+ string(mIt->value.GetString())
		     + "expected :" +typeName );
    }
    DEBUG("Found "<<typeName);      

    DEBUG("Searching for Length");   
    mIt = it->value.FindMember("Length");
    if( mIt == it->value.MemberEnd() ) {
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Length");
        
    
    }

    DEBUG("Found "<< std::stoi(mIt->value.GetString()));      
    size_t length = std::stoi(mIt->value.GetString());
    
    outVector->clear();
    //outVector->resize( std::stoi(mIt->value.GetString()) );
   
    mIt = it->value.FindMember("Members");
    if( mIt == it->value.MemberEnd() ){
      PALISADE_THROW(lbcrypto::deserialize_error, "could not find Members");
    }
    DEBUG("found members");
    const SerialItem& members = mIt->value;
    DEBUG("looping over members");
    //loop over entire vector
    for( size_t i=0; i<length; i++ ) {
      std::string keystring = std::to_string(i);

      //find this key (the index)
      DEBUG(" Searching for "<<keystring);
      Serialized::ConstMemberIterator eIt = members.FindMember(keystring);
      if( eIt == members.MemberEnd() ) {
	PALISADE_THROW(lbcrypto::deserialize_error, "could not find vector entry "+to_string(i));
      }
      DEBUG(" found "<<keystring);

      //within the key's member, find the sub member with the typename
      //and point to it with s2.
      string matrix_name = "Matrix_"+to_string(i);
      DEBUG(" Searching for "<<matrix_name);
      SerialItem::ConstMemberIterator s2 = eIt->value.FindMember(matrix_name);
      if( s2 == eIt->value.MemberEnd() ){
	PALISADE_THROW(lbcrypto::deserialize_error,
		       "could not find matrix name "+ matrix_name);
      }
      DEBUG("Found "<<matrix_name);
      
      // within s2,
      Serialized ser(rapidjson::kObjectType);
      SerialItem k( typeName, ser.GetAllocator() );
      SerialItem v( s2->value, ser.GetAllocator() );
      DEBUGEXP(i);
      if (s2->value.IsString()) {
	DEBUGEXP(s2->value.GetString());
      }
      if (s2->value.IsUint64()){ 
	DEBUGEXP(s2->value.GetUint64());
      }
      ser.AddMember(k, v, ser.GetAllocator());

      //now deserialize the Matrix at in s2
      std::string mat_name = "Matrix";
      //std::string elemname = (outVector->at(i)).GetElementName(); fails for T==BitInt
      std::string elem_name = typeName;
      DEBUG("Calling DeserializeMaatrix");

      //auto pT = make_shared<Matrix<T>>(alloc_function, 0,0);
      auto pT = make_shared<Matrix<T>>([](){ return T(); }, 0,0);
      bool rc = DeserializeMatrix(mat_name, elem_name, s2, pT.get());
      if (rc) {
	DEBUG("Deserialized matrix at index "<<i);
      } else {
	PALISADE_THROW(lbcrypto::deserialize_error, "Deserialization of Matrix "+to_string(i)+" failed internally");
      }
      outVector->push_back(*pT); //store the pointer to the Matrix<T> into the vector location
    }
    return true;
  }

  
  
  // TODO: These functions appear to be used only in
  // benchmark/src/diffSnapshot.cpp they should be documented and
  // possibly moved to another file in utils?
  
  class IStreamWrapper {
  public:
    typedef char Ch;
    
  IStreamWrapper(std::istream& is) : is_(is) {
    }
    
    Ch Peek() const { // 1
      int c = is_.peek();
      return c == std::char_traits<char>::eof() ? '\0' : (Ch)c;
    }
    
    Ch Take() { // 2
      int c = is_.get();
      return c == std::char_traits<char>::eof() ? '\0' : (Ch)c;
    }

    size_t Tell() const { return (size_t)is_.tellg(); } // 3

    Ch* PutBegin() { assert(false); return 0; }
    void Put(Ch) { assert(false); }
    void Flush() { assert(false); }
    size_t PutEnd(Ch*) { assert(false); return 0; }

  private:
    IStreamWrapper(const IStreamWrapper&);
    IStreamWrapper& operator=(const IStreamWrapper&);

    std::istream& is_;
  };

  class OStreamWrapper {
  public:
    typedef char Ch;

  OStreamWrapper(std::ostream& os) : os_(os) {
    }

    Ch Peek() const { assert(false); return '\0'; }
    Ch Take() { assert(false); return '\0'; }
    size_t Tell() const { return 0; }

    Ch* PutBegin() { assert(false); return 0; }
    void Put(Ch c) { os_.put(c); }                  // 1
    void Flush() { os_.flush(); }                   // 2
    size_t PutEnd(Ch*) { assert(false); return 0; }

  private:
    OStreamWrapper(const OStreamWrapper&);
    OStreamWrapper& operator=(const OStreamWrapper&);

    std::ostream& os_;
  };

}

#endif
