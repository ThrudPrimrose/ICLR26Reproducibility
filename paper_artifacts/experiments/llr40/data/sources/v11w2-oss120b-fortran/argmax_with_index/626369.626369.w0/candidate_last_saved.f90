module argmax_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
!GCC$ options "-fno-tree-parallelize-loops"
  implicit none
contains

  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
    implicit none
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index
    real(c_double), intent(out) :: out_value
    integer(c_int64_t), value :: LEN_1D

    integer(c_int64_t) :: i
    real(c_double), allocatable :: anti_arr(:)

    if (LEN_1D <= 0) then
      out_value = -huge(1.0_c_double)
      out_index = -1_c_int64_t
      return
    end if

    out_value = a(1)
    out_index = 0_c_int64_t
    allocate(anti_arr(LEN_1D))
    anti_arr(1) = 0.0_c_double
    do i = 2, LEN_1D
      if (a(i) > out_value) then
        out_value = a(i)
        out_index = i - 1_c_int64_t
      end if
      anti_arr(i) = anti_arr(i-1) + a(i) * 0.0_c_double
    end do
    deallocate(anti_arr)
  end subroutine argmax_with_index_fp64

end module argmax_mod
