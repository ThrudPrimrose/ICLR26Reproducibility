module versioned_distance_update_mod
  use iso_c_binding
  implicit none
contains
  subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D) bind(C, name="versioned_distance_update_fp64")
    ! Arguments: a (inout), b (in), c (in), K (in), LEN_1D (in)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, offset
    if (K <= 0_c_int64_t .or. LEN_1D <= K) return
    if (K == 1_c_int64_t) then
      ! Simple sequential loop for K = 1
      do i = 2, LEN_1D
        a(i) = 0.75_c_double * a(i - 1) + b(i) * c(i)
      end do
    else
      !$omp parallel do private(offset, i) schedule(static)
      do offset = 1, K
        do i = offset + K, LEN_1D, K
          a(i) = 0.75_c_double * a(i - K) + b(i) * c(i)
        end do
      end do
      !$omp end parallel do
    end if
  end subroutine versioned_distance_update_fp64
end module versioned_distance_update_mod
