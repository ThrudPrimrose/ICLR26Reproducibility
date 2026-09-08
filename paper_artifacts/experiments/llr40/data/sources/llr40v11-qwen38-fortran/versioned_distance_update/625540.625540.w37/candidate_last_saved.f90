!> versioned_distance_update: a(i) = 0.75*a(i-K) + b(i)*c(i), i = K+1..N
!> Plain Fortran entry (symbol "versioned_distance_update_"):
!> arrays passed as data pointers, scalars by reference, int64 sizes.
      subroutine versioned_distance_update(a, b, c, len_1d, k)
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int64_t), intent(in) :: len_1d, k
        real(c_double), intent(inout) :: a(len_1d)
        real(c_double), intent(in) :: b(len_1d), c(len_1d)
        integer(c_int64_t) :: i

        do i = k + 1, len_1d
           a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
        end do
      end subroutine versioned_distance_update
