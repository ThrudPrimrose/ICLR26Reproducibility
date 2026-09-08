module versioned_distance_update_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D) bind(C, name="versioned_distance_update_fp64")
    real(c_double), intent(inout) :: a(0:*)
    real(c_double), intent(in) :: b(0:*), c(0:*)
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: offset, i
    if (K <= 0_c_int64_t .or. K >= LEN_1D) return
    !$omp parallel default(none) shared(a,b,c,LEN_1D,K) private(offset,i)
    !$omp do schedule(static)
    do offset = 0_c_int64_t, K - 1_c_int64_t
      do i = K + offset, LEN_1D - 1, K
        a(i) = 0.75_c_double * a(i - K) + b(i) * c(i)
      end do
    end do
    !$omp end parallel
  end subroutine versioned_distance_update_fp64
end module versioned_distance_update_mod
