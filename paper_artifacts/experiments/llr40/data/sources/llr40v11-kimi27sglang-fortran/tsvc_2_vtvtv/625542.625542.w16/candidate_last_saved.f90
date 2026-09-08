module tsvc_2_vtvtv_m
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_loc, c_intptr_t
  implicit none
contains
  subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(c, name='tsvc_2_vtvtv_fp64')
    integer(c_int64_t), value :: len_1d
    real(c_double), target, intent(inout) :: a(len_1d)
    real(c_double), target, intent(in) :: b(len_1d)
    real(c_double), target, intent(in) :: c(len_1d)
    integer(c_intptr_t) :: addr
    addr = transfer(c_loc(a(1_c_int64_t)), addr)
    if (iand(addr, 63_c_intptr_t) == 0_c_intptr_t) then
      call work_aligned(a, b, c, len_1d)
    else
      call work_unaligned(a, b, c, len_1d)
    end if
  contains
    subroutine work_aligned(aa, bb, cc, n)
      integer(c_int64_t), value :: n
      real(c_double), pointer, contiguous :: aa(:), bb(:), cc(:)
      integer(c_int64_t) :: i
!$omp simd aligned(aa,bb,cc:64)
      do i = 1_c_int64_t, n
        aa(i) = aa(i) * bb(i) * cc(i)
      end do
    end subroutine
    subroutine work_unaligned(aa, bb, cc, n)
      integer(c_int64_t), value :: n
      real(c_double), pointer, contiguous :: aa(:), bb(:), cc(:)
      integer(c_int64_t) :: i
!$omp simd
      do i = 1_c_int64_t, n
        aa(i) = aa(i) * bb(i) * cc(i)
      end do
    end subroutine
  end subroutine
end module
