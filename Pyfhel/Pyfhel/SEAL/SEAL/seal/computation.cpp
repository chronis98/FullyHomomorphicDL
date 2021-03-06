#include <stdexcept>
#include "computation.h"

using namespace std;

namespace seal
{
    namespace util
    {
        FreshComputation::FreshComputation(int plain_max_coeff_count, uint64_t plain_max_abs_value) :
            plain_max_coeff_count_(plain_max_coeff_count), plain_max_abs_value_(plain_max_abs_value)
        {
#ifdef SEAL_DEBUG
            if (plain_max_coeff_count <= 0)
            {
                throw invalid_argument("plain_max_coeff_count");
            }
#endif
        }

        FreshComputation::~FreshComputation()
        {
        }

        Simulation FreshComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.get_fresh(parms, plain_max_coeff_count_, plain_max_abs_value_);
        }

        FreshComputation *FreshComputation::clone()
        {
            return new FreshComputation(plain_max_coeff_count_, plain_max_abs_value_);
        }

        AddComputation::AddComputation(Computation &input1, Computation &input2)
        {
            input1_ = input1.clone();
            input2_ = input2.clone();
        }

        AddComputation::~AddComputation()
        {
            delete input1_;
            delete input2_;
        }

        Simulation AddComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.add(input1_->simulate(parms), input2_->simulate(parms));
        }

        AddComputation *AddComputation::clone()
        {
            return new AddComputation(*input1_, *input2_);
        }

        AddManyComputation::AddManyComputation(vector<Computation*> inputs)
        {
#ifdef SEAL_DEBUG
            if (inputs.empty())
            {
                throw invalid_argument("inputs can not be empty");
            }
#endif
            for (size_t i = 0; i < inputs.size(); i++)
            {
#ifdef SEAL_DEBUG
                if (inputs[i] == nullptr)
                {
                    throw invalid_argument("inputs can not contain null pointers");
                }
#endif
                inputs_.emplace_back(inputs[i]->clone());
            }
        }

        AddManyComputation::~AddManyComputation()
        {
            for (size_t i = 0; i < inputs_.size(); i++)
            {
                delete inputs_[i];
            }
        }

        Simulation AddManyComputation::simulate(const EncryptionParameters &parms)
        {
            vector<Simulation> inputs;
            for (size_t i = 0; i < inputs_.size(); i++)
            {
                inputs.emplace_back(inputs_[i]->simulate(parms));
            }
            return simulation_evaluator_.add_many(inputs);
        }

        AddManyComputation *AddManyComputation::clone()
        {
            return new AddManyComputation(inputs_);
        }

        SubComputation::SubComputation(Computation &input1, Computation &input2)
        {
            input1_ = input1.clone();
            input2_ = input2.clone();
        }

        SubComputation::~SubComputation()
        {
            delete input1_;
            delete input2_;
        }

        Simulation SubComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.sub(input1_->simulate(parms), input2_->simulate(parms));
        }

        SubComputation *SubComputation::clone()
        {
            return new SubComputation(*input1_, *input2_);
        }

        MultiplyComputation::MultiplyComputation(Computation &input1, Computation &input2)
        {
            input1_ = input1.clone();
            input2_ = input2.clone();
        }

        MultiplyComputation::~MultiplyComputation()
        {
            delete input1_;
            delete input2_;
        }

        Simulation MultiplyComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.multiply(input1_->simulate(parms), input2_->simulate(parms));
        }

        MultiplyComputation *MultiplyComputation::clone()
        {
            return new MultiplyComputation(*input1_, *input2_);
        }

        RelinearizeComputation::RelinearizeComputation(Computation &input, int decomposition_bit_count) :
            decomposition_bit_count_(decomposition_bit_count)
        {
#ifdef SEAL_DEBUG
            // Check that decomposition_bit_count is in correct interval
            if (decomposition_bit_count <= 0 || decomposition_bit_count > SEAL_DBC_MAX)
            {
                throw invalid_argument("decomposition_bit_count is not in the valid range");
            }
#endif
            input_ = input.clone();
        }

        RelinearizeComputation::~RelinearizeComputation()
        {
            delete input_;
        }

        Simulation RelinearizeComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.relinearize(input_->simulate(parms), decomposition_bit_count_);
        }

        RelinearizeComputation *RelinearizeComputation::clone()
        {
            return new RelinearizeComputation(*input_, decomposition_bit_count_);
        }

        MultiplyPlainComputation::MultiplyPlainComputation(Computation &input, int plain_max_coeff_count, uint64_t plain_max_abs_value) :
            plain_max_coeff_count_(plain_max_coeff_count), plain_max_abs_value_(plain_max_abs_value)
        {
#ifdef SEAL_DEBUG
            if (plain_max_coeff_count <= 0)
            {
                throw invalid_argument("plain_max_coeff_count");
            }
#endif
            input_ = input.clone();
        }

        MultiplyPlainComputation::~MultiplyPlainComputation()
        {
            delete input_;
        }

        Simulation MultiplyPlainComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.multiply_plain(input_->simulate(parms), plain_max_coeff_count_, plain_max_abs_value_);
        }

        MultiplyPlainComputation *MultiplyPlainComputation::clone()
        {
            return new MultiplyPlainComputation(*input_, plain_max_coeff_count_, plain_max_abs_value_);
        }

        AddPlainComputation::AddPlainComputation(Computation &input, int plain_max_coeff_count, uint64_t plain_max_abs_value) :
            plain_max_coeff_count_(plain_max_coeff_count), plain_max_abs_value_(plain_max_abs_value)
        {
#ifdef SEAL_DEBUG
            if (plain_max_coeff_count <= 0)
            {
                throw invalid_argument("plain_max_coeff_count");
            }
#endif
            input_ = input.clone();
        }

        AddPlainComputation::~AddPlainComputation()
        {
            delete input_;
        }

        Simulation AddPlainComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.add_plain(input_->simulate(parms), plain_max_coeff_count_, plain_max_abs_value_);
        }

        AddPlainComputation *AddPlainComputation::clone()
        {
            return new AddPlainComputation(*input_, plain_max_coeff_count_, plain_max_abs_value_);
        }

        SubPlainComputation::SubPlainComputation(Computation &input, int plain_max_coeff_count, uint64_t plain_max_abs_value) :
            plain_max_coeff_count_(plain_max_coeff_count), plain_max_abs_value_(plain_max_abs_value)
        {
#ifdef SEAL_DEBUG
            if (plain_max_coeff_count <= 0)
            {
                throw invalid_argument("plain_max_coeff_count");
            }
#endif
            input_ = input.clone();
        }

        SubPlainComputation::~SubPlainComputation()
        {
            delete input_;
        }

        Simulation SubPlainComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.sub_plain(input_->simulate(parms), plain_max_coeff_count_, plain_max_abs_value_);
        }

        SubPlainComputation *SubPlainComputation::clone()
        {
            return new SubPlainComputation(*input_, plain_max_coeff_count_, plain_max_abs_value_);
        }

        NegateComputation::NegateComputation(Computation &input)
        {
            input_ = input.clone();
        }

        NegateComputation::~NegateComputation()
        {
            delete input_;
        }

        Simulation NegateComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.negate(input_->simulate(parms));
        }

        NegateComputation *NegateComputation::clone()
        {
            return new NegateComputation(*input_);
        }

        ExponentiateComputation::ExponentiateComputation(Computation &input, uint64_t exponent, int decomposition_bit_count) : 
            exponent_(exponent), decomposition_bit_count_(decomposition_bit_count)
        {
#ifdef SEAL_DEBUG
            // Check that decomposition_bit_count is in correct interval
            if (decomposition_bit_count <= 0 || decomposition_bit_count > SEAL_DBC_MAX)
            {
                throw invalid_argument("decomposition_bit_count is not in the valid range");
            }
#endif
            input_ = input.clone();
        }

        ExponentiateComputation::~ExponentiateComputation()
        {
            delete input_;
        }

        Simulation ExponentiateComputation::simulate(const EncryptionParameters &parms)
        {
            return simulation_evaluator_.exponentiate(input_->simulate(parms), exponent_, decomposition_bit_count_);
        }

        ExponentiateComputation *ExponentiateComputation::clone()
        {
            return new ExponentiateComputation(*input_, exponent_, decomposition_bit_count_);
        }

        MultiplyManyComputation::MultiplyManyComputation(vector<Computation*> inputs, int decomposition_bit_count) :
            decomposition_bit_count_(decomposition_bit_count)
        {
#ifdef SEAL_DEBUG
            if (inputs.empty())
            {
                throw invalid_argument("inputs can not be empty");
            }

            // Check that decomposition_bit_count is in correct interval
            if (decomposition_bit_count <= 0 || decomposition_bit_count > SEAL_DBC_MAX)
            {
                throw invalid_argument("decomposition_bit_count is not in the valid range");
            }
#endif
            for (size_t i = 0; i < inputs.size(); i++)
            {
#ifdef SEAL_DEBUG
                if (inputs[i] == nullptr)
                {
                    throw invalid_argument("inputs can not contain null pointers");
                }
#endif
                inputs_.emplace_back(inputs[i]->clone());
            }
        }

        MultiplyManyComputation::~MultiplyManyComputation()
        {
            for (size_t i = 0; i < inputs_.size(); i++)
            {
                delete inputs_[i];
            }
        }

        Simulation MultiplyManyComputation::simulate(const EncryptionParameters &parms)
        {
            vector<Simulation> inputs;
            for (size_t i = 0; i < inputs_.size(); i++)
            {
                inputs.emplace_back(inputs_[i]->simulate(parms));
            }
            return simulation_evaluator_.multiply_many(inputs, decomposition_bit_count_);
        }

        MultiplyManyComputation *MultiplyManyComputation::clone()
        {
            return new MultiplyManyComputation(inputs_, decomposition_bit_count_);
        }
    }
}