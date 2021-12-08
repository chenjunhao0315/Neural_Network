//
//  OtterDType.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/4.
//

#include "OtterDType.hpp"
OtterTypeData* OtterType::typeMetaDatas() {
    static OtterTypeData instances[MaxTypeIndex + 1] = {
    #define SCALAR_TYPE_META(T, name)       \
    /* ScalarType::name */                  \
    OtterTypeData(                           \
        sizeof(T),                          \
        _PickNew<T>(),                      \
        _PickPlacementNew<T>(),             \
        _PickCopy<T>(),                     \
        _PickPlacementDelete<T>(),          \
        _PickDelete<T>()),
    DTYPE_CONVERSION_TABLE(SCALAR_TYPE_META)
    #undef SCALAR_TYPE_META
  };
  return instances;
}

std::string toString(ScalarType type) {
    switch(type) {
#define DEFINE_STR(_1, n) case ScalarType::n: return std::string(#n); break;
        DTYPE_CONVERSION_TABLE(DEFINE_STR)
#undef DEFINE_STR
        default:
            return std::string("Undefined");
    }
}
