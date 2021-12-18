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
    OTTER_ALL_SCALAR_TYPES(SCALAR_TYPE_META)
    #undef SCALAR_TYPE_META
  };
  return instances;
}
