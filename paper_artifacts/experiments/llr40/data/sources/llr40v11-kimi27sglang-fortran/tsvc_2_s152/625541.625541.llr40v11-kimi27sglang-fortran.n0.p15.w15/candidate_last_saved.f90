module tsvc_2_s152_m
  use iso_c_binding, only: c_int64_t, c_double, c_ptr, c_f_pointer
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(c, name='tsvc_2_s152_fp64')
    type(c_ptr), value :: a, b, c, d, e
    integer(c_int64_t), value :: LEN_1D
    real(c_double), pointer, contiguous :: fa(:), fb(:), fc(:), fd(:), fe(:)
    integer(c_int64_t) :: i

    call c_f_pointer(a, fa, [LEN_1D])
    call c_f_pointer(b, fb, [LEN_1D])
    call c_f_pointer(c, fc, [LEN_1D])
    call c_f_pointer(d, fd, [LEN_1D])
    call c_f_pointer(e, fe, [LEN_1D])

    !$omp simd aligned(fa,fb,fc,fd,fe:64)
    do i = 1_c_int64_t, LEN_1D
      fb(i) = fd(i) * fe(i)
      fa(i) = fa(i) + fb(i) * fc(i)
    end do
    !$omp end simd
  end subroutine tsvc_2_s152_fp64
end module tsvc_2_s152_m
