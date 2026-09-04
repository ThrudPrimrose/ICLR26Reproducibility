module tsvc_2_s323_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s323_fp64")
    implicit none
    integer(c_int64_t), value :: len_1d
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in) :: c(*), d(*), e(*)
    integer :: i, n, i_unroll_end
    real(c_double) :: prev_b, cur_a, ci
    n = len_1d
    if (n <= 1) return
    prev_b = b(1)
    ! Unrolled loop processing 8 elements per iteration
    i_unroll_end = n - mod(n - 1, 8)  ! largest index <= n that aligns with groups of 8 starting at 2
    i = 2
    do while (i <= i_unroll_end)
      ! Process eight elements: i, i+1, ..., i+7
      ci = c(i)
      cur_a = prev_b + ci * d(i)
      a(i) = cur_a
      prev_b = cur_a + ci * e(i)
      b(i) = prev_b

      ci = c(i+1)
      cur_a = prev_b + ci * d(i+1)
      a(i+1) = cur_a
      prev_b = cur_a + ci * e(i+1)
      b(i+1) = prev_b

      ci = c(i+2)
      cur_a = prev_b + ci * d(i+2)
      a(i+2) = cur_a
      prev_b = cur_a + ci * e(i+2)
      b(i+2) = prev_b

      ci = c(i+3)
      cur_a = prev_b + ci * d(i+3)
      a(i+3) = cur_a
      prev_b = cur_a + ci * e(i+3)
      b(i+3) = prev_b

      ci = c(i+4)
      cur_a = prev_b + ci * d(i+4)
      a(i+4) = cur_a
      prev_b = cur_a + ci * e(i+4)
      b(i+4) = prev_b

      ci = c(i+5)
      cur_a = prev_b + ci * d(i+5)
      a(i+5) = cur_a
      prev_b = cur_a + ci * e(i+5)
      b(i+5) = prev_b

      ci = c(i+6)
      cur_a = prev_b + ci * d(i+6)
      a(i+6) = cur_a
      prev_b = cur_a + ci * e(i+6)
      b(i+6) = prev_b

      ci = c(i+7)
      cur_a = prev_b + ci * d(i+7)
      a(i+7) = cur_a
      prev_b = cur_a + ci * e(i+7)
      b(i+7) = prev_b

      i = i + 8
    end do
    ! Process remaining elements
    do while (i <= n)
      ci = c(i)
      cur_a = prev_b + ci * d(i)
      a(i) = cur_a
      prev_b = cur_a + ci * e(i)
      b(i) = prev_b
      i = i + 1
    end do
  end subroutine tsvc_2_s323_fp64
end module tsvc_2_s323_mod
